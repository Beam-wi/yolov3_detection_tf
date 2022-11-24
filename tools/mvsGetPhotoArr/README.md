###
You can modify the path of mvs camera take photos log in th utils/cfgs.py
Check error code from utils/ercode

###
Demo as fallow.

***
    import cv2
    import time
    
    from mvs_get_photo_arr import MvsGetPhoto
    from multiprocessing.dummy import Pool as ThreadPool
    
    if __name__ == "__main__":
        # # Get one image.
        # mvsGetPhoto = MvsGetPhoto()
        mvsGetPhoto.camera_init()
        mvsGetPhoto.setMvsCameraParam("黄色")
        # img0 = mvsGetPhoto.get_photo(sn0)
        # img1 = mvsGetPhoto.get_photo(sn1)
        # img2 = mvsGetPhoto.get_photo(sn2)
        # mvsGetPhoto.destroy_handle()
        # mvsGetPhoto.destroy_thread()
    
        # With thread.
        mvsGetPhoto = MvsGetPhoto(thread=True)
        mvsGetPhoto.camera_init()
        mvsGetPhoto.setMvsCameraParam("黄色")
        images = mvsGetPhoto.thread_get_photo([sn1, sn2, ...])
        or serials = mvsGetPhoto.serials
           images = mvsGetPhoto.thread_get_photo(serials)
        mvsGetPhoto.destroy_handle()
        mvsGetPhoto.destroy_thread()