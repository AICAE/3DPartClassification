from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import ui

from bs4 import BeautifulSoup
import os, os.path
import time


driver = webdriver.Chrome("/opt/bin/chromedriver")

import traceparts_parameters as tp

tp.login(driver)
time.sleep(5)  # wait for page can been seen/ready

url="https://www.traceparts.com/en/search/traceparts-classification-mechanical-components-fasteners-screws-and-bolts?CatalogPath=TRACEPARTS%3ATP01001013"
category_level = 4

driver.get(url) # if no such page, fail silently
time.sleep(5)  # wait for page can been seen
driver.maximize_window()
driver.implicitly_wait(5)

tp.apply_filters(driver)
time.sleep(5)  # wait for page can been seen/ready


tree = driver.find_element_by_css_selector(
        "div.result-filter-box > div.treeview-nodes-container")

css_selector = " > ".join(["div.treeview-nodes-container" for l in range(category_level-1)])
#items = tree.find_elements_by_css_selector(css_selector)

#.flex-row.color-font-16 result-filter-item 
items = tree.find_elements_by_css_selector("div.treeview-node-child")
print(len(items))

for item in items:
    print(tp.get_category_full_name(item))
    try:
        print(tp.get_category_url(item))  # all faild
    except:
            print(tp.get_category_link(item))

#driver.quit()
