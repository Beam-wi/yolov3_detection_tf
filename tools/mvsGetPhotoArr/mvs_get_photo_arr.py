# -- coding: utf-8 --

import os
import time
import sys
import numpy as np
import termios

from multiprocessing.dummy import Pool as ThreadPool

sys.path.append(os.path.dirname(__file__))
# from MvCameraControl_class import *
from MvImport.MvCameraControl_class import *
from lib.DbUtils import DbUtil
from lib.logger import get_logger


def press_any_key_exit():
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    # sys.stdout.write(msg)
    # sys.stdout.flush()
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 7)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)


class MvsGetPhoto():
    """
    Take photos from MVS camera.
    """

    def __init__(self, ):
        self.logger = get_logger("MVS_take_photo")
        self.dbUtil = DbUtil()
        self.camera_enum()

    def camera_enum(self, ):
        """
        Enumerate devices get the found serial num.
        :return:-
        """

        self.SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        self.logger.debug("SDKVersion[0x%x]" % self.SDKVersion)

        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        # ch:枚举设备 | en:Enum device
        self.ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
        if self.ret != 0:
            self.logger.error("Enum devices fail! ret[0x%x]" % self.ret)
            raise Exception("Enum devices fail! ret[0x%x]" % self.ret)

            # raise ValueError(
            #     f"Find {self.deviceList.nDeviceNum} device but expect {self.device}.")
        # self.logger.debug("Find %d devices!" % self.deviceList.nDeviceNum)

        # 获取各设备序列号
        self.idevice = self.deviceList.nDeviceNum
        assert self.idevice, "No camera connected, check the status of camera."
        self.camera_sdn = dict()
        self.camera_dsn = dict()
        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(
                self.deviceList.pDeviceInfo[i],
                POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                self.logger.debug("Find gige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                self.logger.debug("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp &
                         0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp &
                         0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp &
                         0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp &
                        0x000000ff)
                self.logger.debug(
                    "current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                ip = f'{nip1}{nip2}{nip3}{nip4}'
                self.camera_sdn[ip] = {'device': i,
                                       'model_name': strModeName,
                                       'ip': ip}
                self.camera_dsn[i] = {'device': i,
                                      'model_name': strModeName,
                                      'ip': ip}
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                self.logger.debug("Find u3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                self.logger.debug("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                self.camera_sdn[strSerialNumber] = {'device': i,
                                                    'model_name': strModeName,
                                                    'serial_num': strSerialNumber}
                self.camera_dsn[i] = {'device': i,
                                      'model_name': strModeName,
                                      'serial_num': strSerialNumber}
                self.logger.debug("user serial number: %s\n" % strSerialNumber)

    def camera_init(self,
                    mfs_file,
                    sn_list=None,
                    thread=True,
                    img_format='RGB',
                    time_out=1000,
                    b_sleep=1,):
        """
        Camera initialize.
        :param mfs_file: The path of mfs file with default parameters.
        :param sn_list: List of serial [sel1, sel2 ...], Default all if Nome.
        :param thread: If you want use multiprocess.
        :param img_format: The picture format that you want to get, BGR or RGB.
        :param time_out: Timeout for get frame from camera, ms.
        :param b_sleep: Sleep time for buffer frame after camera initialize, s.
        """
        assert os.path.exists(mfs_file), f'{mfs_file} not exist.'
        # assert len(
        #     os.listdir(mfs_file)) == self.idevice, f'mfs file number wrong.'

        self.mfs_files = {
            x.split('_')[-1].split('.mfs')[0]: os.path.join(mfs_file, x)
            for x in os.listdir(mfs_file)} if len(os.listdir(mfs_file)) > 0 \
            else {}
        print(self.mfs_files)

        if thread:
            try:
                self.destroy_thread()
            except Exception as e:
                pass
            self.pool = ThreadPool(
                self.idevice)  # 创建n个容量的线程池并发执行
            self.logger.debug(f"Thread has been created successfully.")

        assert img_format in ['RGB',
                              'BGR'], f'Nonsupport this format: {img_format}'
        self.img_format = img_format
        self.PixelType_Gvsp_Packed = {'RGB': PixelType_Gvsp_RGB8_Packed,
                                      'BGR': PixelType_Gvsp_BGR8_Packed}
        self.time_out = time_out
        self.camera_enum()
        if sn_list and self.deviceList.nDeviceNum != len(sn_list):
            self.logger.warning(
                f"Find {self.deviceList.nDeviceNum} device but get {len(sn_list)}.")
        self.init_sn = self.camera_sdn.keys() if not sn_list else sn_list
        try:
            initeds = copy.deepcopy(list(self.ParamDict.keys()))
            if len(initeds) > 0:
                self.logger.warning(f"Handles exist, they will be destroyed.")
                # self.destroy_handle(list(set(self.init_sn) & set(initeds)))
                self.destroy_handle(initeds)
        except Exception as e:
            self.logger.debug(f"No handle exist, it will be created.")

        self.ParamDict = dict()
        for ii, sn in enumerate(self.init_sn):
            assert sn in self.camera_sdn.keys(), \
                f'{sn} not exist in enumeration {list(self.camera_sdn.keys())}'
            self.ParamDict[sn] = dict()
            # ch:创建相机实例 | en:Creat Camera Object
            cam = MvCamera()

            # ch:选择设备并创建句柄| en:Select device and create handle
            stDeviceList = cast(
                self.deviceList.pDeviceInfo[int(self.camera_sdn[sn]['device'])],
                POINTER(MV_CC_DEVICE_INFO)).contents

            ret = cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.logger.error(f"{sn} create handle fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} create handle fail! ret[0x%x]" % ret)

            # ch:打开设备 | en:Open device
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                self.logger.error(f"{sn} open device fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} open device fail! ret[0x%x]" % ret)

            # ch:探测网络最佳包大小(只对GigE相机有效) |
            # en:Detection network optimal package size
            # (It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",
                                                nPacketSize)
                    if ret != 0:
                        self.logger.warning(
                            f"Warning: {sn} set Packet Size fail! ret[0x%x]" % ret)
                else:
                    self.logger.warning(
                        f"Warning: {sn} get Packet Size fail! ret[0x%x]" % nPacketSize)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                self.logger.error(
                    f"{sn} set trigger mode fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} set trigger mode fail! ret[0x%x]" % ret)

            # 设置默认参数 | from mfs. must before set PixelType.
            if sn in self.mfs_files.keys():
                ret = cam.MV_CC_FeatureLoad(self.mfs_files[sn])
                if ret != 0:
                    self.logger.error(f"{sn} set mfs file fail! ret[0x%x]" % ret)
                    raise Exception(f"{sn} set mfs file fail! ret[0x%x]" % ret)

            # 设置曝光参数白平衡参数
            # 前端掉方法设置

            # ch:设置视频流格式 PixelType_Gvsp_RGB8_Packed
            # 或 PixelType_Gvsp_BGR8_Packed(暂时不支持)
            ret = cam.MV_CC_SetEnumValue(
                "PixelFormat", self.PixelType_Gvsp_Packed[self.img_format])
            if ret != 0:
                self.logger.error(f"{sn} set pixelType fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} set pixelType fail! ret[0x%x]" % ret)

            # ch:获取数据包大小| en:Get payload size
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                self.logger.error(
                    f"{sn} get payload size fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} get payload size fail! ret[0x%x]" % ret)
            nPayloadSize = stParam.nCurValue

            # 插入开始
            # ch:获取帧率大小| en:Get the frame rate size
            framrate = MVCC_FLOATVALUE()
            ret = cam.MV_CC_GetFloatValue("ResultingFrameRate", framrate)
            if ret != 0:
                self.logger.error(f"{sn} get framrate fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} get framrate fail! ret[0x%x]" % ret)
            assert not framrate.fCurValue < 6, \
                f"{sn} frame rate {framrate.fCurValue} unequal set 6."

            # ch:开始取流| en:Start grab image
            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                self.logger.error(f"{sn} start grabbing fail! ret[0x%x]" % ret)
                raise Exception(f"{sn} start grabbing fail! ret[0x%x]" % ret)

            stDeviceList = MV_FRAME_OUT_INFO_EX()
            memset(byref(stDeviceList), 0, sizeof(stDeviceList))

            data_buf = (c_ubyte * nPayloadSize)()
            self.ParamDict[sn] = {'cam': cam, 'ret': ret,
                                  'stDeviceList': stDeviceList,
                                  'stParam': stParam,
                                  'nPayloadSize': nPayloadSize,
                                  'data_buf': data_buf}
            self.logger.debug(f"Serial {sn} initialized successfully.")

        time.sleep(b_sleep)

    def get_photo(self, deSn):
        """
        Get one photo from specified camera.
        :param deNum: The num of connection device, type str.
        :return: image
        """
        assert deSn in self.camera_sdn.keys(), f'No this camera serial {deSn}'
        assert isinstance(deSn, str) and deSn in self.ParamDict.keys(), \
            f'{deSn} device has not been initialized.'

        # ch:开始取流| en:Start grab image
        self.ParamDict[deSn]['ret'] = \
            self.ParamDict[deSn]['cam'].MV_CC_GetOneFrameTimeout(
                byref(self.ParamDict[deSn]['data_buf']),
                self.ParamDict[deSn]['nPayloadSize'],
                self.ParamDict[deSn]['stDeviceList'],
                self.time_out)

        if self.ParamDict[deSn]['ret'] == 0:
            try:
                img = np.frombuffer(
                    self.ParamDict[deSn]['data_buf'], dtype=np.uint8).reshape(
                    (self.ParamDict[deSn]['stDeviceList'].nHeight,
                     self.ParamDict[deSn]['stDeviceList'].nWidth, 3))
                self.logger.info(
                    f"{self.camera_sdn[deSn]['model_name']} Serial: {deSn} "
                    f"Device: {self.camera_sdn[deSn]} Shape: {img.shape}")

                # return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return {'sn': deSn, 'img': img}

            except Exception as e:
                self.logger.error(
                    f"Convert file executed failed with error {e}.")
                raise Exception(f"Convert file executed failed with error {e}.")
        else:
            framrate_ = MVCC_FLOATVALUE()
            ret = self.ParamDict[deSn]['cam'].MV_CC_GetFloatValue("ResultingFrameRate", framrate_)
            self.logger.error(f"framrate_: {framrate_.fCurValue}Camera {deSn} get one frame fail, ret[0x%x]" %
                              self.ParamDict[deSn]['ret'])


            raise Exception(f"Camera {deSn} get one frame fail, ret[0x%x]" %
                            self.ParamDict[deSn]['ret'])

    def thread_get_photo(self, devices):
        """
        Get the photo with multithreading.
        :param devices: The list about connection device, type list.
        """
        assert isinstance(devices, list), 'Devices must be list.'
        PR = {x['sn']: x['img'] for x in self.pool.map(self.get_photo, devices)}

        return [PR[sn] for sn in devices]

    def destroy_handle(self, deSns=None):
        """
        Destroy the handle. Not must be executed.
        The camera will lost connection for minutes
        if this function not be executed after grab images.
        :param deNum: The list about connection device or None.
                      Destroy all handle when deNums is None.
        """
        try:
            deSns = copy.deepcopy(list(self.ParamDict.keys())) if not deSns else deSns
        except Exception as e:
            self.logger.error(f'Confirm the camera initialised, {e}')
            raise Exception(f'Confirm the camera initialised, {e}')

        for i, deSN in enumerate(deSns):
            # ch:停止取流 | en:Stop grab image
            self.ParamDict[deSN]['ret'] = self.ParamDict[deSN][
                'cam'].MV_CC_StopGrabbing()
            if self.ParamDict[deSN]['ret'] != 0:
                del self.ParamDict[deSN]['data_buf']
                self.logger.error(
                    "stop grabbing fail! ret[0x%x]" % self.ParamDict[deSN][
                        'ret'])
                raise Exception(
                    "stop grabbing fail! ret[0x%x]" % self.ParamDict[deSN][
                        'ret'])

            # ch:关闭设备 | Close device
            self.ParamDict[deSN]['ret'] = self.ParamDict[deSN][
                'cam'].MV_CC_CloseDevice()
            if self.ParamDict[deSN]['ret'] != 0:
                del self.ParamDict[deSN]['data_buf']
                self.logger.error(
                    "close deivce fail! ret[0x%x]" % self.ParamDict[deSN][
                        'ret'])
                raise Exception(
                    "close deivce fail! ret[0x%x]" % self.ParamDict[deSN][
                        'ret'])

            # ch:销毁句柄| Destroy handle
            self.ParamDict[deSN]['ret'] = self.ParamDict[deSN][
                'cam'].MV_CC_DestroyHandle()
            if self.ParamDict[deSN]['ret'] != 0:
                del self.ParamDict[deSN]['data_buf']
                self.logger.error(
                    "destroy handle fail! ret[0x%x]" % self.ParamDict[deSN][
                        'ret'])
                raise Exception(
                    "destroy handle fail! ret[0x%x]" % self.ParamDict[deSN][
                        'ret'])

            # del self.ParamDict[deSN]['data_buf']
            del self.ParamDict[deSN]
            self.logger.debug(
                f"Serial {deSN} handle has been destroyed successfully.")

    def destroy_thread(self):
        """
        Destroy the thread.
        """
        self.pool.close()  # 关闭线程池，不再接受新的线程
        self.pool.join()  # 主线程阻塞等待子线程的退出
        self.logger.debug(f"Thread has been closed successfully.")

    @property
    def enumSerials(self):
        """
        :return: The serial num of serials which connected. (enumerated)
        """
        try:
            enumserials = copy.deepcopy(list(self.camera_sdn.keys()))
        except Exception as e:
            self.logger.error(f'{e} Confirm camera_enum has not been called.')
            raise Exception(f'{e} Confirm camera_enum has not been called.')
        return enumserials

    @property
    def initSerials(self):
        """
        :return: The serial num of serials which initialized.
        """
        try:
            initserials = copy.deepcopy(list(self.ParamDict.keys()))
        except Exception as e:
            self.logger.error(f'{e} No camera has been initialized.')
            raise Exception(f'{e} No camera has been initialized.')
        return initserials

    def getPositionSN(self):
        """
        :return: The map of dict with position key and sn value.
        And The map of dict with sn key and position value.
        """
        posnMap = {}
        snpoMap = {}
        sql = "select position, sn, canvas_id from tCameraConfig"
        cameraData = self.dbUtil.queryAll(sql, None)
        if cameraData in [(), [], {}]:
            self.logger.error('Nothing from sql server, '
                              'when getPositionSN executed.')
            raise Exception('Nothing from sql server, '
                            'when getPositionSN executed.')
        for cameraOneData in cameraData:
            posnMap[cameraOneData[0]] = cameraOneData[1]
            snpoMap[cameraOneData[1]] = cameraOneData[0]
        return posnMap, snpoMap

    def getCameraonfig(self):
        """
        获得摄像机位置参数
        """
        resultMap = {}
        sql = "select position, sn, canvas_id,camera_group, camera_order " \
              "from tCameraConfig order by camera_order"
        cameraData = self.dbUtil.queryAll(sql, None)
        if cameraData in [(), [], {}]:
            self.logger.error('Nothing from sql server, '
                              'when getCameraonfig executed.')
            raise Exception('Nothing from sql server, '
                            'when getCameraonfig executed.')
        for cameraOneData in cameraData:
            resultMap[cameraOneData[0]] = [cameraOneData[1], cameraOneData[3],
                                           cameraOneData[4]]

        return resultMap

    def getExposureByParamsMark(self, cameraParamMark):
        sql = "select param_json from tCameraParamRecord " \
              "where camera_param_mark = %s"
        params = [cameraParamMark]
        data = self.dbUtil.queryOne(sql, params)
        if data in [(), [], {}]:
            self.logger.error('Nothing from sql server, '
                              'when getExposureByParamsMark executed.')
            raise Exception('Nothing from sql server, '
                            'when getExposureByParamsMark executed.')
        return data[0]

    def setMvsCameraParam(self, cameraParamMark):
        """
        Set the exposure and white balance.
        :param cameraParamMark: 黑色、黄色、白色
        :return: -
        """
        try:
            self.snpoMap
        except Exception as e:
            try:
                self.posnMap, self.snpoMap = self.getPositionSN()
                self.edevice = len(self.snpoMap)
            except Exception as e:
                self.logger.error(f"Accessing database report error. {e}")
                raise Exception(f"Accessing database report error. {e}")
        try:
            deSns = self.ParamDict.keys()
        except Exception as e:
            self.logger.error(f'Confirm the camera initialised, {e}')
            raise Exception(f'Confirm the camera initialised, {e}')
        param_json = self.getExposureByParamsMark(cameraParamMark)
        param_json = eval(param_json)
        self.logger.info(f"Get param from database: {param_json}")
        for i, sn in enumerate(deSns):
            self.logger.info(f"Set camera: {sn}")
            # 关闭自动曝光 | 0:Off 1:Once 2.Continuous
            ret = self.ParamDict[sn]['cam'].MV_CC_SetEnumValue(
                "ExposureAuto", 0)
            if ret != 0:
                self.logger.error(
                    f"{sn} close ExposureAuto fail! ret[0x%x]" % ret)
                raise Exception(
                    f"{sn} close ExposureAuto fail! ret[0x%x]" % ret)

            # 设置自动曝光 | IFloat  ≥0.0，单位us
            ret = self.ParamDict[sn]['cam'].MV_CC_SetFloatValue(
                "ExposureTime",
                float(param_json[str(self.snpoMap[sn])]['ExposureTime']['value']))
            if ret != 0:
                self.logger.error(
                    f"{sn} set ExposureAuto fail! ret[0x%x]" % ret)
                raise Exception(
                    f"{sn} set ExposureAuto fail! ret[0x%x]" % ret)

            # 关闭自动白平衡 | 0:Off 1:Once 2.Continuous
            ret = self.ParamDict[sn]['cam'].MV_CC_SetEnumValue(
                "BalanceWhiteAuto", 0)
            if ret != 0:
                self.logger.error(
                    f"{sn} close BalanceWhiteAuto fail! ret[0x%x]" % ret)
                raise Exception(
                    f"{sn} close BalanceWhiteAuto fail! ret[0x%x]" % ret)

            # 选则通道 BalanceRatioSelector  0: Red 1: Green 2. Blue
            # 修改白平衡值 BalanceRatio ≥0
            for cinfo in param_json[str(self.snpoMap[sn])]['BalanceRatioSelector']:
                ret = self.ParamDict[sn]['cam'].MV_CC_SetEnumValue(
                    "BalanceRatioSelector", int(cinfo['value']))
                if ret != 0:
                    self.logger.error(
                        f"{sn} channel {cinfo['value']} BR select fail! ret[0x%x]" % ret)
                    raise Exception(
                        f"{sn} channel {cinfo['value']} BR select fail! ret[0x%x]" % ret)

                ret = self.ParamDict[sn]['cam'].MV_CC_SetIntValue(
                    "BalanceRatio",
                    int(cinfo['group']['BalanceRatio']['value']))
                if ret != 0:
                    self.logger.error(
                        f"{sn} channel {cinfo['value']} BR set fail! ret[0x%x]" % ret)
                    raise Exception(
                        f"{sn} channel {cinfo['value']} BR set fail! ret[0x%x]" % ret)


"""
if __name__ == "__main__":
    # # Get one image.
    # mvsGetPhoto = MvsGetPhoto()
    mvsGetPhoto.camera_init(mfs_file)
    mvsGetPhoto.setMvsCameraParam("黄色")
    # img0 = mvsGetPhoto.get_photo(sn0)
    # img1 = mvsGetPhoto.get_photo(sn1)
    # img2 = mvsGetPhoto.get_photo(sn2)
    # mvsGetPhoto.destroy_handle()
    # mvsGetPhoto.destroy_thread()

    # With thread.
    mvsGetPhoto = MvsGetPhoto(thread=True)
    mvsGetPhoto.camera_init(mfs_file)
    mvsGetPhoto.setMvsCameraParam("黄色")
    images = mvsGetPhoto.thread_get_photo([sn1, sn2, ...])
    or serials = mvsGetPhoto.serials
       images = mvsGetPhoto.thread_get_photo(serials)
    mvsGetPhoto.destroy_handle()
    mvsGetPhoto.destroy_thread()
"""
