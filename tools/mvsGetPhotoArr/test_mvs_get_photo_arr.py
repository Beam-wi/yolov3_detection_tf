# coding:utf-8

import cv2
import time

from mvs_get_photo_arr import MvsGetPhoto
from multiprocessing.dummy import Pool as ThreadPool

# if __name__ == "__main__":
#     mvsGetPhoto = MvsGetPhoto(4)
#     devices = ["00D59253196", "00E50919334", "00D59253204", "00E41843416"]
#     # mvsGetPhoto.camera_init()
#
#     for i in range(10):
#         start = time.time()
#         a = mvsGetPhoto.thread_get_photo(devices)
#         print(f"lost time: {time.time()-start}\n")
#         time.sleep(1)
#         # print(len(a))
#         # for j, once in enumerate(a):
#         #     print(once.shape)
#         # start1 = time.time()
#         # for j, once in enumerate(a):
#         #     jpg_file = f"/mnt/data/binovo/data/images/AfterConvert_RGB_{j}.jpg"
#         #     cv2.imwrite(jpg_file, once)
#         # print(f"save lost time: {time.time()-start1}")
#
#     mvsGetPhoto.destroy_handle()
#     mvsGetPhoto.destroy_thread()


if __name__ == "__main__":
    mvsGetPhoto = MvsGetPhoto()
    devices = ["00D59253196", "00E50919334", "00D59253204", "00E41843416"]
    print(f"mvsGetPhoto.enumSserials: {mvsGetPhoto.enumSerials}")
    # print(f"mvsGetPhoto.initSerials: {mvsGetPhoto.initSerials}")
    aaa = time.time()
    mvsGetPhoto.camera_init("/opt/config/ai-product-injection-mold-inserts/camera", thread=True)
    print(f"init loss: {time.time()-aaa}")
    # print(f"mvsGetPhoto.initSerials: {mvsGetPhoto.initSerials}")
    bbb = time.time()
    mvsGetPhoto.setMvsCameraParam("黄色")
    print(f"param loss: {time.time()-bbb}")

    for i in range(2):
        start = time.time()
        a = mvsGetPhoto.thread_get_photo(devices)
        print(f"lost time: {time.time()-start}\n")
        time.sleep(1)
        # print(len(a))
        # for j, once in enumerate(a):
        #     print(once.shape)
        # start1 = time.time()
        # for j, once in enumerate(a):
        #     jpg_file = f"/mnt/data/binovo/data/images/AfterConvert_RGB_{j}.jpg"
        #     cv2.imwrite(jpg_file, once)
        # print(f"save lost time: {time.time()-start1}")

    mvsGetPhoto.destroy_handle()
    # mvsGetPhoto.destroy_thread()

    print("-------------------------------------------------")
    # print(f"mvsGetPhoto.enumSerials: {mvsGetPhoto.enumSerials}")
    # print(f"mvsGetPhoto.initSerials: {mvsGetPhoto.initSerials}")
    aaaa = time.time()
    mvsGetPhoto.camera_init(sn_list=["00D59253196", "00E50919334", "00D59253204"])
    print(f"init loss: {time.time()-aaaa}")
    # print(f"mvsGetPhoto.initSerials: {mvsGetPhoto.initSerials}")
    bbbb = time.time()
    mvsGetPhoto.setMvsCameraParam("黑色")
    print(f"param loss: {time.time()-bbbb}")
    for i in range(2):
        start = time.time()
        a = mvsGetPhoto.thread_get_photo(["00D59253196", "00E50919334", "00D59253204"])
        print(f"lost time: {time.time()-start}\n")
        time.sleep(1)
        # print(len(a))
        # for j, once in enumerate(a):
        #     print(once.shape)
        # start1 = time.time()
        # for j, once in enumerate(a):
        #     jpg_file = f"/mnt/data/binovo/data/images/AfterConvert_RGB_{j}.jpg"
        #     cv2.imwrite(jpg_file, once)
        # print(f"save lost time: {time.time()-start1}")

    mvsGetPhoto.destroy_handle()
    mvsGetPhoto.destroy_thread()

