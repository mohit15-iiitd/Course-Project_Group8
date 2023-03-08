
import praw
from prawcore.exceptions import Redirect
from prawcore.exceptions import ResponseException
from urllib.error import HTTPError
import pandas as pd
import os

CLIENT_ID = 'A8RNhFPMwtRdXoGTc46SWA'
SECRET = 'OHJg79eYPTcFOPFUsbgQGQ44J8yF4w'
USER_AGENT = 'iiitdm_8'

reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=SECRET, user_agent=USER_AGENT)

def get_subreddit(tag, num_posts):
    try:
        depression_posts = list(reddit.subreddit(tag).hot(limit=num_posts))

    except Redirect:
            print("Invalid Subreddit!")
            return 0

    except HTTPError:
        print("Too many Requests. Try again later!")
        return 0

    except ResponseException:
        print("Client info is wrong. Check again.")
        return 0
    
    return depression_posts


def get_scraped_data(posts, TEXT_PATH):
    post_title = list()
    post_count = 1

    for post in posts[2:]:
        post_title.append([post_count, post.title, post.selftext])
        post_count += 1

    dataset = pd.DataFrame(post_title, columns=["id", "title", "body"])
    dataset.to_csv(r'{}'.format(TEXT_PATH) + '/excitement_data.csv', index=False)
    return 1


def create_directories(TEXT_PATH):
    parent_path = 'E:\M_TECH ASSIGNMENTS\Information Retrieval\Project'
    permission_mode = 0o777

    text_path = os.path.join(parent_path, TEXT_PATH)

    try:
        os.mkdir(text_path, permission_mode)
        return text_path
    
    except OSError as error: 
        print(error) 


if __name__ == '__main__':
    tag = 'smiling'
    posts = 5002

    TEXT_PATH = 'E:\M_TECH ASSIGNMENTS\Information Retrieval\Project/text_data'

    # text_path = create_directories(TEXT_PATH)
    reddit_posts = get_subreddit(tag, posts)

    get_scraped_data(reddit_posts, TEXT_PATH)
