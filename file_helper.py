import logging
import os

import shutil

import zipfile
import json

import pandas as pd


class FileHelper:

    def __init__(self, zip_path = "tmp"):
        self.this_file_dir  = os.path.dirname(os.path.realpath(__file__))  
        self.coclus_res = os.path.join(
            self.this_file_dir, "res", "khiops_res", "coclus")  
        self.train_test_folder = os.path.join(
            self.this_file_dir, "res", "train_test_splitted")   
        self.transfered_folder = os.path.join(
            self.this_file_dir, "res", "deployed") 
        self.cav_folder = os.path.join(
            self.this_file_dir, "res", "cav")   
        self.tmp_path = os.path.join("/tmp", "code")
        self.ensure_dirs_exist([
            self.tmp_path, self.train_test_folder, 
            self.transfered_folder, self.cav_folder])
        self.zip_path = os.path.join(self.tmp_path, zip_path + ".zip") 
        logging.info("File_helper instantiated")

    def write_days_on_disk(self, days, file_name, label):
        """
        From a pandas, a file name and a label; write days in disk
        in a single csv file. Days are sorted for the sake of khiops.
        """
        path = os.path.join(self.train_test_folder, file_name + label + ".csv")
        fn = self.get_file_name(path)
        days["n_day_"] = days["n_day_"].apply(str)
        days.sort_values(by=['n_day_', "time_"]).drop(["date_"], axis = 1).to_csv(path, sep = ";", index = False)
        logging.info("Wrote the file %s on disk", fn)
        return path

    def write_cav_on_disk(self, cav, file_name, nb_cluster, label):
        out_cav_folder = os.path.join(self.cav_folder, file_name)
        self.ensure_dir_exists(out_cav_folder)
        out_path_json = os.path.join(out_cav_folder, file_name + "_ncluster" + str(nb_cluster) + "_" + label + ".json")
        
        data = {}
        for k, v in cav.items():
            data[str(k)] = str(sorted(v))
            
        with open(out_path_json, "w") as fp:
            json.dump(data, fp, sort_keys = True)

        logging.info("Wrote cav on disk for file %s with nb cluster %s in %s", file_name, nb_cluster, label)

    def write_labels(self, path_data, mcn):
        fn = self.get_file_name(path_data)
        df = pd.read_csv(path_data, sep = ";")
        uq = df["n_day_"].unique()
        df = pd.DataFrame({"n_day_": uq})
        path = os.path.join(self.transfered_folder, fn + "_deployed_"+ str(mcn) + ".csv")
        df["n_day_"] = df["n_day_"].apply(str)
        df.sort_values(by=['n_day_']).to_csv(path, sep = ";", index=False)
        logging.info("Read and transfert file %s to %s", path_data, path)
        return path

    def zip_code(self):
        """
        At the start of each run, save the actual code for reproductible 
        research
        """
        zipf = zipfile.ZipFile(
            self.zip_path, 
            'w', zipfile.ZIP_DEFLATED)
        cfp = self.get_files_paths_in_folder(
            self.this_file_dir, filter_extention = '.py')

        for f in cfp:
            zipf.write(f, "code/" + os.path.basename(f))

        zipf.close()
        logging.info("Saved execution code on disk in zip %s", self.zip_path)

    def zip_data(self, in_dir, res_path):
        """
        At the end of succeeded execution, write data used and results.
        """
        zipf = zipfile.ZipFile(
            self.zip_path, 
            'a', zipfile.ZIP_DEFLATED) # append!
        csv = self.get_files_paths_in_folder(
            in_dir, filter_extention = ".csv")
        res = self.get_files_paths_in_folder(
            res_path)

        for f in csv:
            zipf.write(f, "csv/" + os.path.basename(f))
        
        for f in res:
            zipf.write(f, "res/" + os.path.basename(f))

        zipf.close()
        logging.info("Saved results and data on disk in zip %s", self.zip_path)

    def rm_simplified_outputs(self, file_name, mcn):
        """
        Removed temporary coclustering res file. Keep only the good coclustering
        files.
        """
        for ext in [".json", ".khc"]:
            p = os.path.join(self.coclus_res, file_name, file_name + "_Simplified-" + str(mcn) + ext)
            os.remove(p)

    def clean_zips_folder(self):
        """
        Procedure to clean zip output folder
        remove empty zip files (if I stop execution before the end...)
        """
        files = self.get_files_paths_in_folder(self.tmp_path)
        for f in files:
            s = os.path.getsize(f)
            if s < 35000:
                os.remove(f) 
                logging.info("Removed %s, empty zip file", f)   

    @staticmethod
    def clean_res_folder(out_dir_res):
        """
        Procedure to clean res output folder
        """
        for the_file in os.listdir(out_dir_res):
            file_path = os.path.join(out_dir_res, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    @staticmethod
    def _get_absolute_files_paths(directory):
        """
        Read the content of a folder to find all absolute path of the files 
        in it.
        """
        for dirpath, _ ,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))

    def get_files_paths_in_folder(self, mypath, filter_extention = ''):
        """
        Read the content of a folder to find all absolute path of the files 
        in it. It could filter files based on their extention.
        """
        file_paths = self._get_absolute_files_paths(mypath)
        if filter_extention == '':
            return file_paths
        else:
            return [f for f in file_paths if os.path.splitext(f)[1] == filter_extention]
            
    @staticmethod
    def get_file_name(path):
        filename_w_ext = os.path.basename(path)
        filename, _ = os.path.splitext(filename_w_ext)
        return filename
    
    @staticmethod
    def ensure_dir_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def ensure_dirs_exist(self, list_dir):
        for d in list_dir:
            self.ensure_dir_exists(d)
    
    @staticmethod
    def write_dic_on_disk(dic, out_dir):
        for k, v in dic.items():                       
            pob = os.path.join(out_dir, k + ".csv")  
            logging.info("Wrote dataset %s on disk", pob)
            v.to_csv(
                path_or_buf = pob, 
                index = False, sep = ";")
    
    @staticmethod
    def write_df_on_disk(df, out_dir, c):
        pob = os.path.join(out_dir, c + ".csv")  
        logging.info("Wrote dataset %s on disk", pob)
        df.to_csv(
            path_or_buf = pob, 
            index = False, sep = ";")
