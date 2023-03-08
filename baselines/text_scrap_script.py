
# importing the packages
import praw
import pandas as pd
import os

CLIENT_ID = 'A8RNhFPMwtRdXoGTc46SWA'
SECRET = 'OHJg79eYPTcFOPFUsbgQGQ44J8yF4w'
USER_AGENT = 'iiitdm_8'

# creating the reddit object
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=SECRET, user_agent=USER_AGENT)


# utility method to get the list of subreddits related to a given tag
def get_subreddit(tag, num_posts):
    try:
        depression_posts = list(reddit.subreddit(tag).hot(limit=num_posts))

    except Exception as E:
            print("Invalid Subreddit!")
            return 0
    
    return depression_posts


# methdo to store the scraped data in the .csv file
def get_scraped_data(posts, tag, TEXT_PATH):
    post_title = list()
    post_count = 1

    for post in posts[2:]:
        post_title.append([post_count, post.title, post.selftext])
        post_count += 1

    dataset = pd.DataFrame(post_title, columns=["id", "title", "body"])
    dataset.to_csv(r'{}'.format(TEXT_PATH) + '/{}.csv'.format(tag), index=False)
    return 1

# method to create directory
def create_directories(dirname):
    parent_path = 'E:\M_TECH ASSIGNMENTS\Information Retrieval\Project'
    permission_mode = 0o777

    text_path = os.path.join(parent_path, dirname)

    try:
        os.mkdir(text_path, permission_mode)
        return text_path
    
    except OSError as error: 
        print(error) 


if __name__ == '__main__':
    tag = 'smiling'         # subreddit tag
    posts = 5002            # number of top posts

    # TEXT_PATH = 'E:\M_TECH ASSIGNMENTS\Information Retrieval\Project/text_data'
    TEXT_PATH = create_directories('text_data')
    reddit_posts = get_subreddit(tag, posts)
    get_scraped_data(reddit_posts, tag, TEXT_PATH)
