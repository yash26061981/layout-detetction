import numpy as np
import cv2
import os
import pandas as pd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
from PIL import Image
from skimage.morphology import skeletonize


class Table_Detection:
    def __init__(self):
        self.debug = True
        self.output = 'results/'
        self.xbound = 5
        self.ybound = 5
        self.hscalemin = 45
        self.hscalemax = 46
        self.vscalemin = 45
        self.vscalemax = 46

    def make_dir(self,srcdir):
        if not os.path.exists(srcdir):
            os.makedirs(srcdir)

    def remove_directory_contents(self, path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    # To show the image.
    def show_image(self, in_image):
        cv2.imshow('image', in_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_input_data_from_file(self, document_index_file):
        indexes_in_file = 'Document_Index'
        workbook = pd.ExcelFile(document_index_file).parse(0)
        document_database_index = workbook[indexes_in_file]
        return document_database_index

    def add_boundary_mask(self, bitmap):
        out_bmp = bitmap.copy()
        indices = np.where(bitmap == 255)
        x_cord_unique = np.unique(indices[0])
        y_cord_unique = np.unique(indices[1])
        x_cord_minimum = np.min(x_cord_unique)
        y_cord_minimum = np.min(y_cord_unique)
        x_cord_maximum = np.max(x_cord_unique)
        y_cord_maximum = np.max(y_cord_unique)
        for col in range(y_cord_minimum, y_cord_maximum):
            for row in range(self.xbound):
                out_bmp[x_cord_minimum + row][col] = 255
                out_bmp[x_cord_maximum - row][col] = 255


        for row in range(x_cord_minimum, x_cord_maximum):
            for col in range(self.ybound):
                out_bmp[row][y_cord_minimum + col] = 255
                out_bmp[row][y_cord_maximum - col] = 255
        return out_bmp



    def detect_table(self, image_path, process_image):
        to_process_image = image_path + process_image
        print('Processing : ', to_process_image)
        in_gray_image = cv2.imread(to_process_image, 0)
        in_rgb_image = cv2.imread(to_process_image)
        original_rgb_image = in_rgb_image.copy()
        results_dir = self.output + str(process_image[0:-4])
        self.make_dir(results_dir)
        self.remove_directory_contents(results_dir)
        original = os.path.join(results_dir, process_image)
        cv2.imwrite(original, in_rgb_image)

        #blur = cv2.fastNlMeansDenoising(in_gray_image, None, 10, 21)
        blur = in_gray_image
        bitmap = cv2.adaptiveThreshold(~blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        '''
        hor_bmp = bitmap.copy()
        ver_bmp = bitmap.copy()

        hor_scale = 10
        horizontal_scale = (int)(hor_bmp.shape[1] / hor_scale)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_scale, 1))

        hor_bmp = cv2.erode(hor_bmp, hor_kernel)
        hor_bmp = cv2.dilate(hor_bmp, hor_kernel)
        '''
        hor_bmps = []
        for hor_scale in range(self.hscalemin, self.hscalemax):
            hor_bmp = bitmap.copy()
            horizontal_scale = (int)(hor_bmp.shape[1] / hor_scale)
            hor_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (horizontal_scale, 1))

            hor_bmp = cv2.erode(hor_bmp, hor_kernel)
            hor_bmp = cv2.dilate(hor_bmp, hor_kernel)
            hor_bmps.append(hor_bmp)

        hor_bmp = hor_bmps[0]
        for bmps in hor_bmps:
            hor_bmp = cv2.add(hor_bmp, bmps)


        #ver_scale = 5
        ver_bmps = []
        for ver_scale in range(self.vscalemin, self.vscalemax):
            ver_bmp = bitmap.copy()
            vertical_scale = (int)(ver_bmp.shape[0] / ver_scale)
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, vertical_scale))

            ver_bmp = cv2.erode(ver_bmp, ver_kernel)
            ver_bmp = cv2.dilate(ver_bmp, ver_kernel)
            ver_bmps.append(ver_bmp)

        ver_bmp = ver_bmps[0]
        for bmps in ver_bmps:
            ver_bmp = cv2.add(ver_bmp, bmps)

        self.show_image(in_rgb_image)

        mask = cv2.add(hor_bmp, ver_bmp)
        self.show_image(mask)
        conn_mask = self.get_connected_mask(mask)
        self.show_image(conn_mask)
        #mask_skel = skeletonize(mask/255)
        #mask_skel = self.get_open_end_points(mask)
        #mask_hm = self.get_hit_miss_image(mask)
        #self.show_image(mask_hm)
        #img_hm, mask_skel_new = self.get_broken_cord(mask_skel)
        #self.show_image(img_hm)
        #self.show_image(mask_skel_new)
        #joints = cv2.bitwise_and(hor_bmp, ver_bmp)

        #new_mask = self.add_boundary_mask(mask_skel_new)
        #self.show_image(new_mask)

        #img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, hierarchy = cv2.findContours(conn_mask, 1, cv2.CHAIN_APPROX_SIMPLE)

        index = 0
        for cnt in contours:

            area = cv2.contourArea(cnt)
            print('block - %d, area = %f'%(index, area))
            if area < 500:
                continue
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            hull = cv2.convexHull(cnt)

            #joint_roi = joints[x:x + w, y:y + h]
            #_, j_contours, _ = cv2.findContours(joint_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #if len(j_contours) <= 2:
            #    continue
            roi = original_rgb_image[y:y+h, x:x+w]
            if index > 0:
                localised = os.path.join(results_dir, ('block_' + str(index) + '.png'))
                cv2.imwrite(localised, roi)
                cv2.rectangle(in_rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1

        self.show_image(in_rgb_image)
        localised = os.path.join(results_dir, 'localised_blocks.png')
        cv2.imwrite(localised, in_rgb_image)
        localised = os.path.join(results_dir, 'mask.png')
        cv2.imwrite(localised, mask)
        localised = os.path.join(results_dir, 'connmask.png')
        cv2.imwrite(localised, conn_mask)
        '''
        localised = os.path.join(results_dir, 'mask_skel.png')
        cv2.imwrite(localised, mask_skel)
        localised = os.path.join(results_dir, 'mask_connected.png')
        cv2.imwrite(localised, mask_skel_new)
        localised = os.path.join(results_dir, 'mask_img_hm.png')
        cv2.imwrite(localised, img_hm)
        '''
        print('Done')

    def get_connected_mask(self, mask):
        h_kernel = np.ones((1,10), np.uint8)
        v_kernel = np.ones((10, 1), np.uint8)
        d_im_h = cv2.dilate(mask, h_kernel, iterations=1)
        e_im_h = cv2.erode(d_im_h, h_kernel, iterations=1)
        hor = self.get_open_end_points(e_im_h)
        d_im_v = cv2.dilate(mask, v_kernel, iterations=1)
        e_im_v = cv2.erode(d_im_v, v_kernel, iterations=1)

        ver = self.get_open_end_points(e_im_v)
        new_mask = cv2.add(hor, ver)
        return new_mask

    def get_open_end_points(self, mask):
        mask_skel = skeletonize(mask / 255)
        img = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.uint8)
        img.fill(0)

        indices = np.where(mask_skel == True)
        x_cord = indices[0]
        y_cord = indices[1]
        for index in range(0, len(indices[0])):
            img[x_cord[index]][y_cord[index]] = 255
        return img

    def get_hit_miss_image(self, mask):
        mask_skel = skeletonize(mask / 255)
        img = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.uint8)
        img.fill(0)

        indices = np.where(mask_skel == True)
        x_cord = indices[0]
        y_cord = indices[1]
        for index in range(0, len(indices[0])):
            img[x_cord[index]][y_cord[index]] = 255

        kernel = np.array(([0, 1, 0], [1, -1, 1], [0, 1, 0]), dtype='int')
        mask_hm = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        #self.show_image(mask_hm)
        return mask_hm

    def get_broken_cord(self, mask_skel):
        (rows,cols) = np.nonzero(mask_skel)
        img = np.zeros([mask_skel.shape[0], mask_skel.shape[1]], dtype=np.uint8)
        img.fill(0)
        skel_cords = []
        for r,c in zip(rows,cols):
            (col_neigh, row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
            pix_neigh = mask_skel[row_neigh,col_neigh].ravel() != 0
            if np.sum(pix_neigh) == 2:
                skel_cords.append((r,c))
                img[r][c] = 255

        merge_cords = []
        for indx1 in range(len(skel_cords)):
            cord1 = skel_cords[indx1]
            for indx2 in range((indx1+1), len(skel_cords)):
                cord2 = skel_cords[indx2]

                dist = np.sqrt(np.square(cord1[0]-cord2[0]) + np.square(cord1[1]-cord2[1]))
                print('distance between cords (%d,%d) and (%d,%d) is %f'%(cord1[0],cord1[1],cord2[0],cord2[1],dist))
                if dist < 50:
                    addr=[]
                    if cord1[0] < cord2[0]:
                        addr = [cord1[0], cord2[0]]
                    else:
                        addr = [cord2[0], cord1[0]]
                    addc= []
                    if cord1[1] < cord2[1]:
                        addc = [cord1[1], cord2[1]]
                    else:
                        addc = [cord2[1], cord1[1]]
                    merge_cords.append((addr,addc))

        for cords in merge_cords:
            for row in range(cords[0][0], (cords[0][1] + 1)):
                for col in range(cords[1][0], (cords[1][1] + 1)):
                    mask_skel[row][col] = 255


        return img, mask_skel

    def detect_table_from_database(self, document_database, document_index_file):
        document_index = self.get_input_data_from_file(document_index_file)
        self.make_dir(self.output)
        self.remove_directory_contents(self.output)
        for index in range(len(document_index)):
            to_process_image = document_database + '/' + str(document_index[index])
            print('Processing : ', to_process_image)
            in_gray_image = cv2.imread(to_process_image,0)
            in_rgb_image = cv2.imread(to_process_image)
            original_rgb_image = in_rgb_image.copy()

            bitmap = cv2.adaptiveThreshold(~in_gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

            hor_bmp = bitmap.copy()
            ver_bmp = bitmap.copy()

            scale = 15
            horizontal_scale = (int)(hor_bmp.shape[1] / scale)
            hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_scale, 1))

            hor_bmp = cv2.erode(hor_bmp, hor_kernel)
            hor_bmp = cv2.dilate(hor_bmp, hor_kernel)

            vertical_scale = (int)(ver_bmp.shape[0] / scale)
            ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_scale))

            ver_bmp = cv2.erode(ver_bmp, ver_kernel)
            ver_bmp = cv2.dilate(ver_bmp, ver_kernel)

            mask = cv2.add(hor_bmp, ver_bmp)

            joints = cv2.bitwise_and(hor_bmp, ver_bmp)

            img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                print(area)
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon,True)
                x,y,w,h = cv2.boundingRect(approx)

                joint_roi = joints[x:x + w, y:y + h]
                _, j_contours, _ = cv2.findContours(joint_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if len(j_contours) <= 4:
                    continue
                cv2.rectangle(in_rgb_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            in_image_name = self.output + str(document_index[index])
            cv2.imwrite(in_image_name, original_rgb_image)
            orig_document_name = document_index[index]
            extracted_image_name = self.output + str(orig_document_name[0:-4]) + '_extracted.png'
            cv2.imwrite(extracted_image_name, in_rgb_image)
        print('Done')

    def detect_remove_table(self, image_path, to_process_image):

        print('Processing : ', to_process_image)
        in_gray_image = cv2.imread(image_path + to_process_image,0)
        in_rgb_image = cv2.imread(image_path + to_process_image)
        original_rgb_image = in_rgb_image.copy()

        bitmap = cv2.adaptiveThreshold(~in_gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        hor_bmp = bitmap.copy()
        ver_bmp = bitmap.copy()

        scale = 50
        horizontal_scale = scale #(int)(hor_bmp.shape[1] / scale)
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_scale, 1))

        hor_bmp = cv2.erode(hor_bmp, hor_kernel)
        hor_bmp = cv2.dilate(hor_bmp, hor_kernel)

        vertical_scale = scale #(int)(ver_bmp.shape[0] / scale)
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_scale))

        ver_bmp = cv2.erode(ver_bmp, ver_kernel)
        ver_bmp = cv2.dilate(ver_bmp, ver_kernel)

        mask = cv2.add(hor_bmp, ver_bmp)

        without_table_bmp = ~(bitmap - mask)
        without_table_gray_image = cv2.multiply(without_table_bmp, in_gray_image)

        extracted_image_name = self.output + str(to_process_image[0:-4]) + '_extracted.png'
        cv2.imwrite(extracted_image_name, without_table_gray_image)

        img = Image.open(extracted_image_name)
        file_write_name = self.output + str(to_process_image[0:-4]) + '_ocr.txt'
        file1 = open(file_write_name, "w")
        text = pytesseract.image_to_string(img)
        file1.write("".join(str(v) for v in text))
        file1.write('\n')
        print(text)


if __name__ == "__main__":
    table_detection = Table_Detection()
    #table_detection.output = 'E:/Test/sample_invoices/'
    '''
    document_database = 'E:/Test/sample_invoices'
    document_idex_file = 'E:/Test/sample_invoices/document_index.xlsx'
    table_detection.detect_table_from_database(document_database, document_idex_file)
    '''
    image_path = 'E:/Test/sample_invoices/'
    process_image = 'e5invoice-1-638_modify.jpg' #'houseng-invoice-template.png'
    #table_detection.detect_remove_table(image_path, process_image)
    table_detection.detect_table(image_path ,process_image)

    print('Done')

