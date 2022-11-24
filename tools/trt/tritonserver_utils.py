#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from PIL import Image
import sys
import time
import struct
import rapidjson as json

import grpc

from tritonclient import utils
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.utils.shared_memory as shm
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import InferenceServerException
from google.protobuf.json_format import MessageToJson


def model_dtype_to_np(model_dtype):
    if model_dtype == "BOOL":
        return np.bool, "bool"
    elif model_dtype == "INT8":
        return np.int8, "int8"
    elif model_dtype == "INT16":
        return np.int16, "int16"
    elif model_dtype == "INT32":
        return np.int32, "int32"
    elif model_dtype == "INT64":
        return np.int64, "int64"
    elif model_dtype == "UINT8":
        return np.uint8, "uint8"
    elif model_dtype == "UINT16":
        return np.uint16, "uint16"
    elif model_dtype == "FP16":
        return np.float16, "float16"
    elif model_dtype == "FP32":
        return np.float32, "float32"
    elif model_dtype == "FP64":
        return np.float64, "float64"
    elif model_dtype == "BYTES":
        return np.dtype(object), "bytes"
    return None


def get_error_grpc(rpc_error):
    return InferenceServerException(
        msg=rpc_error.details(),
        status=str(rpc_error.code()),
        debug_details=rpc_error.debug_error_string())


def raise_error_grpc(rpc_error):
    raise get_error_grpc(rpc_error) from None


def raise_error(msg):
    """
    Raise error with the provided message
    """
    raise InferenceServerException(msg=msg) from None


def deserialize_bytes_tensor(encoded_tensor, datatype):
    offset = 0
    val_buf = encoded_tensor
    # if (datatype != np.object) and (datatype != np.bytes_):
    if datatype in [np.float32, np.float16, np.int32, np.int16, np.int8]:

        return np.frombuffer(val_buf, dtype=datatype, offset=offset)
    else:
        strs = list()
        while offset < len(val_buf):
            l = struct.unpack_from("<I", val_buf, offset)[0]
            offset += 4
            sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
            offset += l
            strs.append(sb)

        return (np.array(strs, dtype=datatype))


def multiplyList(myList):
    result = 1
    for x in myList:
        result = result * abs(x)
    return result


class TritonClient():
    def __init__(self,
                 model_name,
                 model_version='',
                 url='localhost:8001',
                 streaming=False,
                 async_set=False,
                 batch_size=1,
                 scaling='NONE',
                 shared_memory=False,
                 verbose=False,
                 sm_size=512,
                 out_vector=False):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.streaming = streaming
        self.async_set = async_set
        self.batch_size = batch_size
        self.scaling = scaling
        self.shared_memory = shared_memory
        self._verbose = verbose
        # Create gRPC stub for communicating with the server
        self.options = [
            ('grpc.max_message_length', sm_size * sm_size * sm_size),
            ('grpc.max_send_message_length', sm_size * sm_size * sm_size),
            ('grpc.max_receive_message_length', sm_size * sm_size * sm_size)]
        self.out_vector = out_vector
        self.channel = grpc.insecure_channel(self.url, options=self.options)
        self.grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(self.channel)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self.metadata_request = service_pb2.ModelMetadataRequest(
            name=self.model_name, version=self.model_version)
        self.metadata_response = self.grpc_stub.ModelMetadata(
            self.metadata_request)

        self.config_request = service_pb2.ModelConfigRequest(
            name=self.model_name, version=self.model_version)
        self.config_response = self.grpc_stub.ModelConfig(self.config_request)

        self.input_name, self.output_name, self.c, self.h, self.w, self.format, \
        self.i_dtype, self.o_dtype, self.max_batch_size = self.parse_model(
            self.metadata_response, self.config_response.config)

        if shared_memory:
            self.input, self.output_list = self.shcreate_input_output()
            self.shm_ip_handle, self.op_handle_dict = self.create_shared_memory()
        else:
            self.input, self.output_list = self.create_input_output()

    def parse_model(self, model_metadata, model_config):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata.inputs)))

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config.input)))

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata_ns = list()
        output_dtypes = dict()
        for output_metadata in model_metadata.outputs:
            # if output_metadata.datatype != "FP32":
            #     raise Exception(
            #         "expecting output datatype to be FP32, model '" +
            #         model_metadata.name + "' output type is " +
            #         output_metadata.datatype)

            # Output is expected to be a vector. But allow any number of
            # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
            # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
            # is one.
            output_batch_dim = (model_config.max_batch_size > 0)
            non_one_cnt = 0
            for dim in output_metadata.shape:
                if output_batch_dim:
                    output_batch_dim = False
                elif dim > 1:
                    non_one_cnt += 1
                    if self.out_vector and non_one_cnt > 1:
                        raise Exception("expecting model output to be a vector")
            output_metadata_ns.append(
                [output_metadata.name, output_metadata.shape])
            output_dtypes[output_metadata.name] = output_metadata.datatype

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                    format(expected_input_dims, model_metadata.name,
                           len(input_metadata.shape)))

        if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
                (input_config.format != mc.ModelInput.FORMAT_NHWC)):
            raise Exception("unexpected input format " +
                            mc.ModelInput.Format.Name(input_config.format) +
                            ", expecting " +
                            mc.ModelInput.Format.Name(
                                mc.ModelInput.FORMAT_NCHW) +
                            " or " +
                            mc.ModelInput.Format.Name(
                                mc.ModelInput.FORMAT_NHWC))

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]
        input_metadata_ns = [input_metadata.name, input_metadata.shape]

        max_batch_size = model_config.max_batch_size

        return (input_metadata_ns, output_metadata_ns, c, h, w,
                input_config.format, input_metadata.datatype,
                output_dtypes, max_batch_size)

    def create_input_output(self):
        def make_output(x):
            output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = x[-1][0]
            # output.parameters['classification'].int64_param = multiplyList(x[-1])

            return output

        output_list = list(
            map(lambda x: make_output(x),
                enumerate(self.output_name)))

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = self.input_name[0]
        input.datatype = self.i_dtype
        if self.format == mc.ModelInput.FORMAT_NHWC:
            input.shape.extend([self.batch_size, self.h, self.w, self.c])
        else:
            input.shape.extend([self.batch_size, self.c, self.h, self.w])

        return input, output_list

    def shcreate_input_output(self):
        def make_output(x, batch_size):
            output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = x[-1][0]
            # output.parameters['classification'].int64_param = multiplyList(x[-1])
            output = self.set_output_shared_memory(output, f"output{x[0]}_data",
                                                   batch_size * multiplyList(
                                                       x[-1][-1]) * 4)

            return output

        output_list = list(map(lambda x: make_output(x, self.batch_size),
                               enumerate(self.output_name)))

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = self.input_name[0]
        input.datatype = self.i_dtype
        if self.format == mc.ModelInput.FORMAT_NHWC:
            input.shape.extend([self.batch_size, self.h, self.w, self.c])
        else:
            input.shape.extend([self.batch_size, self.c, self.h, self.w])

        input = self.set_input_shared_memory(input, "input_data",
                                             self.batch_size * multiplyList(
                                                 self.input_name[-1]) * 4)

        return input, output_list

    def preprocess(self, img, format, dtype, c, h, w, scaling):
        """
        Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        scaling choices from ['NONE', 'INCEPTION', 'VGG']
        """
        # np.set_printoptions(threshold='nan')

        if c == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        resized_img = sample_img.resize((w, h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        npdtype = model_dtype_to_np(dtype)[0]
        typed = resized.astype(npdtype)

        if scaling == 'INCEPTION':
            scaled = (typed / 128) - 1
        elif scaling == 'VGG':
            if c == 1:
                scaled = typed - np.asarray((128,), dtype=npdtype)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
        else:
            scaled = typed

        # Swap to CHW if necessary
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

        # Channels are in RGB order. Currently model configuration data
        # doesn't provide any information as to other channel orderings
        # (like BGR) so we just assume RGB.
        return ordered

    def postprocess(self, response):
        """
        Post-process response to show classifications.
        """
        assert len(
            response.raw_output_contents) > 0, 'Wrong without result return.'

        return {response.outputs[i].name: np.reshape(
            deserialize_bytes_tensor(raw_output_contents,
            model_dtype_to_np(self.o_dtype[response.outputs[i].name])[0]),
            response.outputs[i].shape) for i, raw_output_contents in
            enumerate(response.raw_output_contents)}

    def shpostprocess(self, response):
        """
        Post-process response to show classifications.
        """
        assert len(
            response.raw_output_contents) > 0, 'Wrong without result return.'

        return {output.name: shm.get_contents_as_numpy(
            self.op_handle_dict[output.name],
            utils.triton_to_np_dtype(self.o_dtype[output.name]),
            output.shape) for i, output in enumerate(response.outputs)}

    def requestGenerator(self, image_data):
        """
        Generate a request.
        You can add parameters['classification'].int64_param for output if you
        want result with classification format.
        """
        assert isinstance(image_data, np.ndarray) and len(
            image_data.shape) == 4, "Input wrong with {}".format(
            image_data)

        request = service_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.model_version = self.model_version

        request.outputs.extend(self.output_list)

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        image_idx = 0
        last_request = False
        while not last_request:
            input_bytes = None
            request.ClearField("inputs")
            request.ClearField("raw_input_contents")
            for idx in range(self.batch_size):
                if input_bytes is None:
                    input_bytes = image_data[image_idx].tobytes()
                else:
                    input_bytes += image_data[image_idx].tobytes()

                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            request.inputs.extend([self.input])
            request.raw_input_contents.extend([input_bytes])
            yield request

    def shrequestGenerator(self, image_data):
        """
        Generate a request.
        You can add parameters['classification'].int64_param for output if you
        want result with classification format.
        """
        assert isinstance(image_data, np.ndarray) and len(
            image_data.shape) == 4, "Input wrong with {}".format(
            image_data)

        request = service_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.model_version = self.model_version

        request.outputs.extend(self.output_list)

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        image_idx = 0
        last_request = False
        while not last_request:
            input_bytes = []
            request.ClearField("inputs")
            request.ClearField("raw_input_contents")
            # for idx in range(self.batch_size):
            for idx in range(image_data.shape[0]):
                input_bytes.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True
            # Put input data values into shared memory
            shm.set_shared_memory_region(self.shm_ip_handle, input_bytes)
            request.inputs.extend([self.input])
            yield request

    def create_shared_memory(self):
        # To make sure no shared memory regions are registered with the
        # server.
        self.unregister_system_shared_memory()
        self.unregister_cuda_shared_memory()
        # Create Output0 and Output1 in Shared Memory
        # and store shared memory handles
        # Register Output0 and Output1 shared memory with Triton Server
        op_handle_dict = dict()
        for i, output in enumerate(self.output_name):
            op_handle_dict[output[0]] = shm.create_shared_memory_region(
                f"output{i}_data",
                f"/output{i}_simple",
                self.batch_size * multiplyList(output[-1]) * 4)
            self.register_system_shared_memory(f"output{i}_data",
                                               f"/output{i}_simple",
                                               self.batch_size * multiplyList(
                                                   output[-1]) * 4)

        # Create Input0 and Input1 in Shared Memory
        # and store shared memory handles
        # Register Input0 and Input1 shared memory with Triton Server
        shm_ip_handle = shm.create_shared_memory_region("input_data",
                                                        "/input_simple",
                                                        self.batch_size * multiplyList(
                                                            self.input_name[
                                                                -1]) * 4)
        self.register_system_shared_memory("input_data", "/input_simple",
                                           self.batch_size * multiplyList(
                                               self.input_name[-1]) * 4)

        return shm_ip_handle, op_handle_dict

    def register_system_shared_memory(self,
                                      name,
                                      key,
                                      byte_size,
                                      offset=0,
                                      headers=None):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        key : str
            The key of the underlying memory object that contains the
            system shared memory region.
        byte_size : int
            The size of the system shared memory region, in bytes.
        offset : int
            Offset, in bytes, within the underlying memory object to
            the start of the system shared memory region. The default
            value is zero.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to register the specified system shared memory.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.SystemSharedMemoryRegisterRequest(
                name=name, key=key, offset=offset, byte_size=byte_size)
            if self._verbose:
                print("register_system_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self.grpc_stub.SystemSharedMemoryRegister(request=request,
                                                      metadata=metadata)
            if self._verbose:
                print("Registered system shared memory with name '{}'".format(
                    name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unregister_system_shared_memory(self, name="", headers=None):
        """Request the server to unregister a system shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the system shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified system shared memory region.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.SystemSharedMemoryUnregisterRequest(name=name)
            if self._verbose:
                print("unregister_system_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self.grpc_stub.SystemSharedMemoryUnregister(request=request,
                                                        metadata=metadata)
            if self._verbose:
                if name is not "":
                    print("Unregistered system shared memory with name '{}'".
                          format(name))
                else:
                    print("Unregistered all system shared memory regions")
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unregister_cuda_shared_memory(self, name="", headers=None):
        """Request the server to unregister a cuda shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the cuda shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified cuda shared memory region.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.CudaSharedMemoryUnregisterRequest(name=name)
            if self._verbose:
                print("unregister_cuda_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self.grpc_stub.CudaSharedMemoryUnregister(request=request,
                                                      metadata=metadata)
            if self._verbose:
                if name is not "":
                    print(
                        "Unregistered cuda shared memory with name '{}'".format(
                            name))
                else:
                    print("Unregistered all cuda shared memory regions")
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_system_shared_memory_status(self,
                                        region_name="",
                                        headers=None,
                                        as_json=False):
        """Request system shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active system shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns system shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or SystemSharedMemoryStatusResponse message holding
            the system shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.SystemSharedMemoryStatusRequest(
                name=region_name)
            if self._verbose:
                print("get_system_shared_memory_status, metadata {}\n{}".format(
                    metadata, request))
            response = self.grpc_stub.SystemSharedMemoryStatus(
                request=request, metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def set_output_shared_memory(self, output, region_name, byte_size,
                                 offset=0):
        """Marks the output to return the inference result in
        specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region to hold tensor data.
        byte_size : int
            The size of the shared memory region to hold tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        Raises
        ------
        InferenceServerException
            If failed to set shared memory for the tensor.
        """
        if 'classification' in output.parameters:
            raise_error("shared memory can't be set on classification output")

        output.parameters[
            'shared_memory_region'].string_param = region_name
        output.parameters[
            'shared_memory_byte_size'].int64_param = byte_size
        if offset != 0:
            output.parameters['shared_memory_offset'].int64_param = offset
        return output

    def set_input_shared_memory(self, input, region_name, byte_size, offset=0):
        """Set the tensor data from the specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        """
        input.ClearField("contents")
        self._raw_content = None

        input.parameters[
            'shared_memory_region'].string_param = region_name
        input.parameters[
            'shared_memory_byte_size'].int64_param = byte_size
        if offset != 0:
            input.parameters['shared_memory_offset'].int64_param = offset
        return input

    def inference(self, image_data):
        assert image_data.dtype == model_dtype_to_np(self.i_dtype)[
            -1], "Expect {}, but get {}".format(
            model_dtype_to_np(self.i_dtype)[-1], image_data.dtype)
        # assert image_data.shape == self.input.shape, \
        #     "With wrong shape of {} expect {}".format(image_data.shape,
        #                                               self.input.shape)
        if image_data.shape[0] != self.batch_size:
            self.batch_size = image_data.shape[0]
            if self.shared_memory:
                self.input, self.output_list = self.shcreate_input_output()
                self.shm_ip_handle, self.op_handle_dict = self.create_shared_memory()
            else:
                self.input, self.output_list = self.create_input_output()

        requests, responses = list(), list()
        # Send request
        if not self.shared_memory:
            if self.streaming:
                for response in self.grpc_stub.ModelStreamInfer(
                        self.requestGenerator(image_data)):
                    responses.append(response)
            else:
                for request in self.requestGenerator(image_data):
                    if not self.async_set:
                        responses.append(self.grpc_stub.ModelInfer(request))
                    else:
                        requests.append(
                            self.grpc_stub.ModelInfer.future(request))

            # For async, retrieve results according to the send order
            if self.async_set:
                for request in requests:
                    responses.append(request.result())

            for response in responses:
                if self.streaming:
                    if response.error_message != "":
                        print(response.error_message)
                        sys.exit(1)
                    else:
                        return self.postprocess(response.infer_response)
                else:
                    return self.postprocess(response)
        else:
            if self.streaming:
                for response in self.grpc_stub.ModelStreamInfer(
                        self.shrequestGenerator(image_data)):
                    responses.append(response)
            else:
                for request in self.shrequestGenerator(image_data):
                    if not self.async_set:
                        responses.append(self.grpc_stub.ModelInfer(request))
                    else:
                        requests.append(
                            self.grpc_stub.ModelInfer.future(request))

            # For async, retrieve results according to the send order
            if self.async_set:
                for request in requests:
                    responses.append(request.result())

            for response in responses:
                if self.streaming:
                    if response.error_message != "":
                        print(response.error_message)
                        sys.exit(1)
                    else:
                        return self.shpostprocess(response.infer_response)
                else:
                    return self.shpostprocess(response)


"""
You can inference any tensorrt model with this script. 
The following demo.
if __name__ == "__main__":
    '''
    model_name: The model name of what you want.
    batch_image_data: A batch image data with type np.float.
    Other args look from the class.
    '''
    tritonclient = TritonClient(model_name)
    data = tritonclient.inference(batch_image_data)

Activate tritonserver with shell command.
#!/usr/bin/env python3

import os


os.system('/opt/tritonserver/tritonserver/install/bin/tritonserver \
--model-repository=~path/model_repository/model_dir')

"""
