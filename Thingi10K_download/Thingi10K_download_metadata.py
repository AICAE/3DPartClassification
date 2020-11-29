"""extract dynamic content
"""



import os.path
import json
import time
import shutil

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from selenium.webdriver import Firefox as Browser
from selenium.webdriver.firefox.options import Options

#from selenium.webdriver.chrome.options import Options

# MOZ_HEADLESS=1 python this_script
options = Options()
options.headless = True
options.binary = shutil.which("firefox")

# chromedriver is per version
#~ chrome_options = Options()  
#~ chrome_options.add_argument("--headless")  
#~ #chrome_options.binary_location =  ""



user_name = "YOUR EMAILID"
password = "YOUR PASSWORD"
driver = Browser(executable_path="/usr/bin/geckodriver", options=options) 
def restart_driver():
    global driver
    driver.quit()
    driver = Browser(executable_path="/usr/bin/geckodriver", options=options)

file_count = 0
restart_count = 20  # close the driver (all tabs)
# https://stackoverflow.com/questions/29502255/is-there-a-way-to-close-a-tab-in-webdriver-or-protractor


def download_metadata(fileid, outputdir):
    output_filename = outputdir + os.path.sep + str(fileid) + ".json"
    metadata = extract_metadata(fileid)
    if not metadata:
        print("metadata is emptry for ", fileid)
    with open(output_filename, "w") as wf:
        json.dump(metadata, wf, indent=4)
    # need to close the tab
    global file_count
    file_count += 1
    if file_count % restart_count == 0:
        restart_driver()
    print("download_metadata file ", output_filename)

def extract_metadata(fileid):
    url = f"https://ten-thousand-models.appspot.com/detail.html?file_id={fileid}"
    body = get_dynamic_content(url)
    info_table = body.find_element_by_css_selector("div#info_panel > table > tbody") # 
    geom_table = body.find_element_by_css_selector("div#geom_panel > table > tbody")
    results = {}
    info = extract_table(info_table, results)
    geom = extract_table(geom_table, results)

    return results

def get_dynamic_content(url):
    driver.get(url)
    driver.implicitly_wait(3) # seconds  # wait for page can been seen, dynamic content loaded
    try:
        body= driver.find_element_by_tag_name("body")
        return body
    except:
        driver.implicitly_wait(10) # seconds
        body= driver.find_element_by_tag_name("body")
        return body

def extract_table(tbody, results = {}):
    for row in tbody.find_elements_by_tag_name("tr"):
        tds = row.find_elements_by_tag_name("td")
        key = tds[0].text
        results[key] = tds[2].text
    return results

if __name__ == "__main__":
    fileid = 34785
    print(extract_metadata(fileid))
