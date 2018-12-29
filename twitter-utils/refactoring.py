import twitter

from datetime import datetime

from credentials import CREDENTIALS

# dataset files path
RAW_NER_FILE = "../dataset/goldnerfile.tsv"
TWEETS_FILE = "../dataset/tweets.tsv"
# polished NER file default separator
SEPARATOR = " > "

api = twitter.Api(consumer_key        = CREDENTIALS["consumer_key"],
                  consumer_secret     = CREDENTIALS["consumer_secret"],
                  access_token_key    = CREDENTIALS["access_token_key"],
                  access_token_secret = CREDENTIALS["access_token_secret"])


if __name__ == "__main__" :
    # loading the NER file
    ner_dictionary = dict()
    with open(RAW_NER_FILE) as ner_file:
        for ner in ner_file.readlines():
            ner = ner.split("\n")[0]
            ner_dictionary[ner[0:18]] = ner
        ner_file.close()
    
    # initializing status dictionary
    status_dictionary = dict()
    # initializing time span dictionary
    timespan = dict()

    starting = len(ner_dictionary.keys())
    counter  = 0
    deleting = 0
    for status in ner_dictionary.keys():
        try:
            # getting tweet
            tweet = api.GetStatus(status).AsDict()
            # storing tweet text
            text  = tweet["text"]
            status_dictionary[status] = text
            print(counter, "->", text)
            counter += 1
            # getting the tweet date
            datetime_object = datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
            datetime_string = str(datetime_object.month) + " " + str(datetime_object.year)
            if datetime_string not in timespan:
                timespan[datetime_string] = 1
            else:
                timespan[datetime_string] += 1
    
        except twitter.error.TwitterError:
            counter  += 1
            deleting += 1
    
    # DEBUG - refactoring
    print("TWEETS: ", starting, "\n", "DELETED: ", deleting, "NOW: ", len(status_dictionary.keys()))
    # DEBUG - tweets timespan
    tspan = ""
    for time in timespan:
        tspan += time + " -> " + str(timespan[time])
        tspan += "\n"
    print("TIMESPAN:\n" + tspan)

    # printing on output file
    with open(TWEETS_FILE, 'w') as file:
        wr = ""
        for status in status_dictionary:
            wr += (status + SEPARATOR + status_dictionary[status] + "\n")
        file.write(wr)
        file.close() 