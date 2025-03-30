from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from selenium.webdriver.common.action_chains import ActionChains

def handle_security_warning(driver):
    # Add your security warning handling logic here
    pass

# Initialize WebDriver with options to handle new tabs
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Step 1: Navigate to the base URL
print("\nStep 1: Navigating to SignBank...")
driver.get("https://signbank.libras.ufsc.br/pt")

# Step 2: Handle any security warnings
print("\nStep 2: Handling security warnings...")
handle_security_warning(driver)

# Step 3: Manual intervention
print("\nStep 3: Manual Intervention Required")
print("Please follow these steps:")
print("1. Click through any remaining security warnings if they appear")
print("2. Navigate to the search page by clicking the necessary buttons")
print("3. The final URL should be: https://signbank.libras.ufsc.br/pt/search-signs/words?page=1&letter=a")
print("\nOnce you have reached the search page, press Enter to scrape the label.")

# Wait for user input
input("Press Enter to continue...")

# Create lists to store data
video_urls = []
sign_labels = []
alphabet_labels = []

# Store the main window handle
main_window = driver.current_window_handle

# Function to get total pages for current category
def get_total_pages(driver):
    try:
        # Wait for pagination elements to load
        time.sleep(2)
        pagination = driver.find_elements(By.XPATH, '//*[@id="page-search-signs-words"]/div[3]/div[2]/nav/ul/li')
        if pagination:
            # The second to last element is the last page number (last element is 'next' button)
            last_page = int(pagination[-2].text)
            return last_page
        return 1
    except Exception as e:
        print(f"Error getting total pages: {e}")
        return 1

# Function to process a single page
def process_page(driver, page_number):
    try:
        # If not on page 1, click the correct page number
        if page_number > 1:
            try:
                # Construct XPath for the specific page number button
                page_button_xpath = f'//*[@id="page-search-signs-words"]/div[3]/div[2]/nav/ul/li[{page_number + 1}]/button'
                page_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, page_button_xpath))
                )
                page_button.click()
                time.sleep(2)
            except Exception as e:
                print(f"Error clicking page {page_number} button: {e}")
                return

        # Get the alphabet label for the current page
        alphabet_label_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[4]/div/div/div[1]/span'))
        )
        current_alphabet_label = alphabet_label_element.text
        
        # Get all sign labels on the current page
        sign_label_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sign-item-card"]/div[2]/span'))
        )
        current_page_labels = [elem.text for elem in sign_label_elements]
        
        # Get all video links on the current page
        video_buttons = driver.find_elements(By.XPATH, '//*[@id="sign-item-card"]/div[2]/a')
        
        # Process each video on the current page
        for i, (video_button, sign_label) in enumerate(zip(video_buttons, current_page_labels), 1):
            try:
                print(f"Processing video {i} on page {page_number}")
                
                # Open link in new tab
                actions = ActionChains(driver)
                actions.key_down(Keys.CONTROL).click(video_button).key_up(Keys.CONTROL).perform()
                
                # Wait for new tab to open and switch to it
                time.sleep(2)
                new_tab = [window for window in driver.window_handles if window != main_window][0]
                driver.switch_to.window(new_tab)
                
                # Wait for video element and get URL
                try:
                    video_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="signs-full-card"]/div[2]/div/div[2]/div[2]/div/div/video/source'))
                    )
                    video_url = video_element.get_attribute("src")
                    # Remove the timestamp part from the URL if present
                    video_url = video_url.split('#')[0]
                    
                    # Store all data
                    video_urls.append(video_url)
                    sign_labels.append(sign_label)
                    alphabet_labels.append(current_alphabet_label)
                    
                except Exception as e:
                    print(f"Error getting video URL: {e}")
                    video_urls.append("URL_NOT_FOUND")
                    sign_labels.append(sign_label)
                    alphabet_labels.append(current_alphabet_label)
                
                # Close current tab and switch back to main window
                driver.close()
                driver.switch_to.window(main_window)
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing video {i} on page {page_number}: {e}")
                if main_window in driver.window_handles:
                    driver.switch_to.window(main_window)
                continue

    except Exception as e:
        print(f"Error processing page {page_number}: {e}")

# Process all alphabetical categories (A to Z, excluding W)
alphabet_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
# Dictionary mapping letters to their XPath positions
letter_xpath_map = {
    'A': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[2]',
    'B': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[3]',
    'C': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[4]',
    'D': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[5]',
    'E': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[6]',
    'F': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[7]',
    'G': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[8]',
    'H': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[9]',
    'I': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[10]',
    'J': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[11]',
    'K': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[12]',
    'L': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[13]',
    'M': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[14]',
    'N': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[15]',
    'O': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[16]',
    'P': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[17]',
    'Q': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[18]',
    'R': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[19]',
    'S': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[20]',
    'T': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[21]',
    'U': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[22]',
    'V': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[23]',
    'X': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[24]',
    'Y': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[25]',
    'Z': '//*[@id="page-search-signs-words"]/div[3]/div[1]/div[3]/div/div/div[2]/div/div[26]'
}

for category in alphabet_categories:
    print(f"\nProcessing category: {category}")
    
    if category != 'A':  # Skip for A since we're already there
        try:
            # Click the category button using the correct XPath
            category_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, letter_xpath_map[category]))
            )
            category_button.click()
            time.sleep(2)
        except Exception as e:
            print(f"Error clicking category {category}: {e}")
            continue
    
    # Get total pages for this category
    total_pages = get_total_pages(driver)
    print(f"Found {total_pages} pages for category {category}")
    
    # Process each page in the category
    for page in range(1, total_pages + 1):
        print(f"Processing page {page} of {total_pages} in category {category}")
        process_page(driver, page)
        print(f"Completed page {page}. Total entries collected so far: {len(video_urls)}")

# Save all data to CSV file
with open("SignBank_metadata.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Alphabet Label", "Sign Label", "Video URL"])  # Header row
    for alphabet, sign, url in zip(alphabet_labels, sign_labels, video_urls):
        writer.writerow([alphabet, sign, url])

# Close WebDriver
driver.quit()

print(f"\nData has been saved to 'video_urls.csv'. Total entries collected: {len(video_urls)}")
