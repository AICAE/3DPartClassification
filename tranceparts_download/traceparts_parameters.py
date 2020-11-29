"""
metadata *.txt  is CSV with double quotes: "Symbol";"Value";"Unit";
"""

import os.path
import json

ISO_only=True
fasteners_only=True
popup_processed=False # will be set in the first download

download_folder="/home/qxia/Downloads/"  # Permission denied
nb_file_per_subcategory = 2

# need to be backup each time, resume to download 
output_folder = "/opt/traceparts_library"

url_website = "https://www.traceparts.com/"
url_login="https://www.traceparts.com/en/sign-in"
url_base="https://www.traceparts.com/en/search/traceparts-classification-mechanical-components-fasteners"
# the suffix is different for different       get from a dict

# ISO standard, then limited to 39 thousand,  estimated to use 5~39GB space
# <input  value="WithCAD" type="checkbox">
# <input  value="ISO" type="checkbox">,
# div.result-filter-items-container>
filters=[       "div.result-filter-items-container>div>input[value=\"WithCAD\"]"
]
#   not always has such filter           "div>input[value=\"ISO\"]"


def apply_filters(driver):
    # after apply filter, parts are listed instead of folder
    for filter in filters:
        driver.find_element_by_css_selector(filter).click()

def login(driver):
    # todo:
    driver.get (url_login)
    driver.find_element_by_id('Email').send_keys('qingfeng.xia@hotmail.com')
    driver.find_element_by_id ('Password').send_keys('Parts2020')
    driver.find_element_by_id ('autoLogin').click()
    driver.find_element_by_id('signin-btn').click()

    try:
        driver.find_element_by_id("cookie-consent-agree").click()
    except:
        pass

def url_to_categories(url):
    pass


"""parse the category tree from "treeview-nodes-container
there are 4 level div has CSS class `treeview-nodes-container` to reach "Screws and bolts"
The topmost level is "Classification", which does not shown in title
it corresponds to <span title = "Mechanical components > Fasteners > Screws and bolts"  ...>
all category levels are available but may not shown in the page

```html
<span title="Mechanical components > Fasteners > Screws and bolts" class="treeview-node-title txt-elip">
<a rel="nofollow" href="/en/search/traceparts-classification-mechanical-components-fasteners-screws-and-bolts?CatalogPath=TRACEPARTS%3ATP01001013" onclick="SearchTreeViewActionApply()">Screws and bolts</a>
</span>
```

"""
def parse_tree(driver, url, start_level):
    subs = {}
    doc = driver.get(url)
    tree = driver.find_element_by_css_selector(
            "div.result-filter-box > div.treeview-nodes-container")

    assert start_level>=2
    for l in range(start_level-1):
        tree = tree.find_element_by_css_selector("div.treeview-nodes-container")

    subs = parse_category(tree)

    #save to json
    with open(output_folder +"category.json", "w") as f:
        f.write(json.dumps(subs, sort_keys=True, indent=4))

def get_category_name(element):
    # div>span.treeview-node-title.txt-elip
    span= element.find_element_by_css_selector("div>span.treeview-node-title.txt-elip")
    span_title = span.get_attribute("title")
    name = span_title.split(">")[-1].strip()
    return name

def get_category_by_name(element, name):
    items = element.find_elements_by_css_selector("div.treeview-nodes-container")
    for item in items:
        span= element.find_element_by_css_selector("div>span.treeview-node-title.txt-elip")
        span_text = span.text
        if span_text.find(name)>0:
            return item

def get_category_full_name(element):
    # div>span.treeview-node-title.txt-elip
    span= element.find_element_by_css_selector("div>span.treeview-node-title.txt-elip")
    return span.get_attribute("title")

def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.
 
Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.
 
"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename

def get_filename_from_full_name(full_name):
    # , > /
    part_name = full_name.split(">")[-1].strip()
    return part_name.replace("/", "-or-")

def get_category_url(element):
    # after applying WithCAD filter
    """
    <button id="99dd537cc091344a8e4c12ff85b0cca0" form="searchForm" type="submit" 
        name="FilterAction-SetCatalogPath" value="TRACEPARTS:TP01001008001"
        formaction="/en/search/traceparts-classification-mechanical-components-fasteners-pins-clevis-pins?CatalogPath=TRACEPARTS%3ATP01001008001"
        formnovalidate="formnovalidate" 
        onclick="setCustomEvent(1,'SearchPage','CategoryFilter-Click','Filter on a classification category');SearchFilterActionApply(this);" 
        class="hidden">
    </button>
    """
    button = element.find_element_by_css_selector("div>span.treeview-node-title.txt-elip>button")
    return url_website + button.get_attribute("formaction")

def get_category_link(element):
    # assume element is last level of div.treeview-nodes-container
    """  before applying the WithCAD filter
    """
    link = element.find_element_by_css_selector("div>span>a")
    return url_website + link.get_attribute("href")

def get_subcategories_count(element):
    try:
        _ret = element.find_element_by_css_selector("div.treeview-nodes-container")
        return len(_ret)
    except:
        return 0

def has_subcategories(element):
    try:
        _ret = element.find_element_by_css_selector("label.treeview-plus")
        return True
    except:
        return False

def parse_category(element):
    # how to limited to the same path depth
    subs={}
    #gname = get_category_name(element)
    gname = get_category_full_name(element)

    items = element.find_elements_by_css_selector("div.treeview-node-child")
    print("category name: ", gname, "len(items) = ", len(items))
    for item in items:
        subs[get_category_full_name(item)] = get_category_url(item)
    return subs

#########################################################
def get_result_items(driver):
    result = driver.find_element_by_css_selector("#search-result-items")
    return result.find_elements_by_css_selector("div.result-content.flex-col")

def get_part_name(element):
    return element.find_element_by_css_selector("div.result-title.txt-elip.color-font-37").text
    
def get_download_button(element):
        """
        <div onclick="setCustomEvent(1,'SearchPage','DownloadPopup-Click','Click to open the CAD download popup in the search result page');" class="tp-button result-partnumber-download-button color-background-5 color-font-3" title="CAD download">
        section#search-result-items> 
        """
        download_button = element.find_element_by_css_selector('div[title="CAD download"]')
        return download_button

def select_cad_format(popup):
    #dropbox list
    popup.find_element_by_css_selector("select#cad-format-select").send_keys("STEP AP214")
    # do not ask again, checkbox
    popup.find_element_by_css_selector("input#cad-choice").click()
    #"button#direct-download"
    popup.find_element_by_css_selector("button#direct-download").click()
    popup.find_element_by_css_selector("div.close-button.tp-button").click()

def get_result_number(driver, result_section):
    """auto paging is enable, needs to scroll down to get more result
    <span title="39,965&nbsp;results">39,965&nbsp;results</span>
    """
    selector = "section#search-result-items> #result-nb-items"
    driver.find_element_by_css_selector(selector)


#launch url
max_level = 5
if fasteners_only:
    registry_filename=output_folder + os.path.sep + "fasteners.json"
    category_list = ["traceparts-classification", "mechanical-components", "fasteners"]
    if ISO_only:
        subcategories = {
            "Screws and bolts": "-screws-and-bolts?CatalogPath=TRACEPARTS%3ATP01001013",
            "Nuts": "-nuts?CatalogPath=TRACEPARTS%3ATP01001007",
            "Washers":"-washers?CatalogPath=TRACEPARTS%3ATP01001017",
            "Rivets": "-rivets?CatalogPath=TRACEPARTS%3ATP01001012",
            "Pins":"-pins?CatalogPath=TRACEPARTS%3ATP01001008"
        }

    def categories_to_url(name):
        #traceparts-classification-mechanical-components-fasteners-screws-and-bolts-cap-screws-countersunk-or-flat-screws?
        #names = [name.lower().replace(" ", "-") for name in categories]
        #return url_base+ "-".join(names) + url_suffix
        url_suffix = subcategories[name]
        return url_base+ url_suffix

else:
    category_list = ["traceparts-classification", "mechanical-components"]
    registry_filename=output_folder + os.path.sep + "mechanical-components.json"
    subcategories="""Fasteners
Bearings
Brakes, clutches and couplings
Linear and rotary motion
Power transmission
Casters, wheels, handling trolleys
Jig and fixture construction
Handles
Hinges, latches & locks
Sealing
Shock/vibration absorbers, springs
Solenoids, Electromagnets""".split("\n")

