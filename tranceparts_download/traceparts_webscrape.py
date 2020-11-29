"""
This is support login with password and check download file
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import ui



from bs4 import BeautifulSoup
import re
#import pandas as pd
import os
import time

from collections import OrderedDict
import sys, glob, os.path
import shutil
import json
import copy
import zipfile
from threading import Thread, Condition

#sudo pip3 install watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def page_is_loaded(driver):
    return driver.find_element_by_tag_name("body") != None

#from traceparts_parameters import url_base, subcategories, registry_filename
#from traceparts_parameters import *
import traceparts_parameters as tp

"""
headless mode to listen to music:  https://realpython.com/modern-web-automation-with-python-and-selenium/#test-driving-a-headless-browser


Unable to locate element: {"method":"css selector","selector":"div[title="CAD download"]"}
"""

cvlock = Condition()
DOWNLOADED_FILENAME=None

class DownloadEventHandler(FileSystemEventHandler): 
    #
    @staticmethod
    def on_created(event): 
        if event.is_directory: 
            return None
  
        if event.event_type == 'created': 
            global DOWNLOADED_FILENAME
            # Event is created, you can process it now 
            if event.src_path[-3:] == "zip":
                print("Watchdog received created event - % s." % event.src_path)
                cvlock.acquire()
                while True:
                    if not DOWNLOADED_FILENAME:
                        DOWNLOADED_FILENAME = event.src_path
                        cvlock.notify_all()
                        break
                    else:
                        cvlock.wait()
                cvlock.release()


class PartDownloader(object):

    def __init__(self, registry_filename):
        self.PART_REGISTRY=OrderedDict()
        if os.path.exists(registry_filename):
            with open(registry_filename) as json_file:
                self.PART_REGISTRY = json.load(json_file)

        self.registry_filename = registry_filename

    def setup(self):
        #read and backup registry_filename

        if True:
            options = webdriver.ChromeOptions()
            # https://chromedriver.chromium.org/capabilities
            options.add_argument("download.default_directory={}".format(tp.download_folder)) # Set the download Path
            
            #DesiredCapabilities capabilities = webdriver.DesiredCapabilities()
            #capabilities.setCapability(webdriver.ChromeOptions.CAPABILITY, options)
            #options.merge(capabilities)  # not tested code copy from Java
            
            # create a new Chrome session, give full path to `chromedriver` or put it on PATH
            self.driver = webdriver.Chrome("/opt/bin/chromedriver", options=options)
        else:  # Firefox brower
            profile =webdriver.FirefoxProfile()
            profile['browser.download.folderList'] = 2 # custom location
            profile['browser.download.dir'] = tp.download_folder
            profile['browser.helperApps.neverAsk.saveToDisk'] = "text/csv,application/pdf,application/zip"
            self.driver = webdriver.Firefox("/opt/bin/geckodriver", profile=profile)
            pass

        wait = ui.WebDriverWait(self.driver, 10)
        wait.until(page_is_loaded)
        self.driver.maximize_window()
        self.driver.implicitly_wait(5)  # wait for javascript or other visual eleemnt ready

        tp.login(self.driver)
        #cookie: do not show download dialog any longer
        # https://towardsdatascience.com/controlling-the-web-with-python-6fceb22c5f08
        self.filename_set = self.get_filename_set()

        self.watch()

    def shutdown(self):
        # 
        self.observer.stop()
        self.observer.join()

        with open(self.registry_filename, 'w') as f:
            f.write(json.dumps(list(self.PART_REGISTRY.values()), indent=4))
            print("data has been write into file: ", self.registry_filename)
        #end the Selenium browser session
        self.driver.quit()

    def setup_cookie():
        #todo: copy or extract cookie from webbrower
        #cookie = {"name" : "foo", "value" : "bar"}
        #self.driver.add_cookie(cookie)
        pass

    def download(self):
        for cat in tp.subcategories:
            url = tp.categories_to_url(cat)
            print(url)
            self.download_subcategory(cat, url)

    def parse_category_tree(self):
        url = tp.categories_to_url(tp.category_list)
        tp.parse_tree(self.driver,url, len(tp.category_list))

    def download_subcategory(self, category_name, url):
        # check if this has been downloaded
        # threading,  also sleep, to simulate human download gap
        #my_categories = copy.copy(tp.category_list)
        #my_categories.append(subcategory)
        #url = tp.categories_to_url(my_categories)
        #print(url)
        self.driver.get(url) # if no such page, fail silently
        time.sleep(3)  # wait for page can been seen
        tp.apply_filters(self.driver)
        time.sleep(8)  # wait for page can been seen/ready

        tree = self.driver.find_element_by_css_selector(
                "div.result-filter-box > div.treeview-nodes-container")
        
        #parse subcategories into a list
        subs = tp.parse_category(tree)
        with open(tp.output_folder + os.path.sep + category_name + ".json", "w") as wf:
            wf.write(json.dumps(subs, indent=4))
        #for name, url in subs.items():
        #    self.download_files(name, url)

    def download_files(self, name, url):
        self.driver.get(url) # if no such page, fail silently
        time.sleep(5)  # wait for page can been seen
        tp.apply_filters(self.driver)
        time.sleep(5)  # wait for page can been seen/ready

        result_items = tp.get_result_items(self.driver)
        if len(result_items)>tp.nb_file_per_subcategory:
            for i in [0, len(result_items)-1]:  # todo:  show at most 40 items

                self.current_part_name = tp.get_filename_from_full_name(name) + "_" +str(i)
                self.current_full_name = name
                print("self.current_part_name", self.current_part_name)
                if self.current_part_name in self.PART_REGISTRY:
                    continue
                #popup nees to deal with once
                if not tp.popup_processed:
                    if True:
                        try:
                            tp.get_download_button(result_items[i]).click()
                            popup = self.driver.find_element_by_css_selector("div#fast-download-window")
                            tp.select_cad_format(popup)
                            tp.popup_processed = True
                        except Exception as e:
                            print(e)
                            #sys.exit(1)  # stop to debug
                    else:  # give longer wait time and manually deal with popup
                        time.sleep(45)
                        tp.popup_processed = True
                else:
                    tp.get_download_button(result_items[i]).click()
                    #takes times to download, result is zip folder, with step file + text meta data
                    time.sleep(1)
                # collect the file and process file
                self.collect()

    def watch(self):
        # not working
        self.observer = Observer()
        event_handler = DownloadEventHandler()
        # can use FileSystemEventHandler class directly,  handler.on_created = 
        self.observer.schedule(event_handler, tp.download_folder, recursive=True)
        self.observer.start()  # in another thread?

    def get_filename_set(self):
        return set(os.listdir(tp.download_folder))

    def collect(self):
        global DOWNLOADED_FILENAME
        collected_filename = None

        #i=0
        #while i<5:
            #time.sleep(1)
            #i+=1

        cvlock.acquire()
        while True:
            if DOWNLOADED_FILENAME:
                collected_filename = DOWNLOADED_FILENAME
                print(collected_filename)
                self.process_file(collected_filename)
                DOWNLOADED_FILENAME = None
                cvlock.notify_all()
                break
            else:
                cvlock.wait()
        cvlock.release()


    def process_file(self, filename):
        """register this file for resuming download"""

        #~ with zipfile.ZipFile(filename) as z:
            #~ for f in z.namelist():
                #~ if f.find(".stp")>0:
                    #~ with z.open(f, "r") as zf:
                        #~ with open(tf.output_folder + os.path.sep + f, 'w') as wf:
                            #~ shutil.copyfileobj(zf, f)
                            #~ d = {"filename": tf.output_folder + os.path.sep + f,
                                     #~ "categories": None }  # todo
        result_file = tp.output_folder + os.path.sep + self.current_part_name + ".zip"
        shutil.copyfile(DOWNLOADED_FILENAME, result_file)
        self.PART_REGISTRY[self.current_part_name] = self.current_full_name
        #do not delete by program
        print(filename)

if __name__ == "__main__":
    downloader = PartDownloader(tp.registry_filename)
    downloader.setup()
    #downloader.parse_category_tree()
    downloader.download()
    downloader.shutdown()