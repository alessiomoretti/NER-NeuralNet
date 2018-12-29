"""
PyPedia is a Python client for DBPedia Lookup APIs (by default the entrypoint is the DBPedia one)
or an equivalent API. Please note that this is simply referred to the Keyword Search.

This is the main script which cover PyPedia utilities, all of them built as a bundle
inside the PyPedia class. This simple library was built to be used in this NER project

So long and thanks for all the fish.
"""

import requests


class PyPedia:
    def __init__(self, api = "http://lookup.dbpedia.org/api/search/KeywordSearch"):
        
        self.api = api

        # api parameters defintion
        self.query_string  = "QueryString"
        self.query_class   = "QueryClass"
        self.max_hits      = "MaxHits"

        # default dict of labels-to-class rules
        # TODO finish to write down rules
        self.classes = {
            "PER" : "person",
            "LOC" : "place",
            "ORG" : "agent",
            "PROD": "thing",
            "ENT" : "thing"
        }

    
    def set_classes(self, labels_dictionary):
        """
        Use this method to overwrite the default labels translation
        into DBPedia ontology classes.
        :param labels_dictionary: dictionary, the dict of labels-to-class rules
        :return: None
        """
        self.classes = labels_dictionary

    def get_class(self, label):
        """
        Use this method to get the ontologically correct DBPedia class
        according to your own rules.abs
        :param label: string
        :return: string
        """
        try:
            return self.classes[label]
        except:
            return ""


    def make_request(self, query, object_class = "", hits = 1):
        """
        This is the method that can be used to perform the actual request
        it return a raw unicode text or None in case of exception.
        :param query: list, the word(s) list to be searched as keyword(s)
        :param object_class: (optional) string, the dbpedia ontological classification
        :param hits: (optional) integer, the maximum number of hits to be returned
        :return: string, the response body - None in case of exception
        """
        # set response accepting JSON 
        headers = {"Accept" : "application/json"}
        # preparing the query (multiple words are supported)
        qstring = "%20".join(query)
        # preparing request parameters
        params  = "?" + self.max_hits     + "=" + str(hits)
        params += "&" + self.query_class  + "=" + object_class
        params += "&" + self.query_string + "=" + qstring

        # performing request
        try:
            r = requests.get(self.api + params, headers=headers)
            return r.text
        except Exception:
            return None

    def prepare_and_request(self, sentence, hits=1):
        """
        This routine can be used to perform a series of request depending upon the 
        list of labelled words returned by the NER system.
        :param sentence: list, the [(word,label), ... ]list of tuple to be recognized as entities or not
        :return: dictionary, the dict of actual results
        """
        to_search = dict()
        for (w, l) in sentence:
            # preparing dictionary to search from in dbpedia
            if len(l) > 1:                          # not 'O'
                [ind, label] = l.split("-")
                if label not in to_search:
                    to_search[label] = [[w]]
                else:
                    label_ = to_search[label]
                    if ind is not "B":              # not begin of entity
                        label_[len(label_) -1 ].append(w)
                    else:
                        label_.append([w])

        results = dict()
        for label in to_search:
            for entity in to_search[label]:
                res = self.make_request(entity, self.classes[label], hits)
                res = self.get_results(res)
                if res is not None:
                    results[" ".join(entity)] = res

        return results
            

    @staticmethod
    def get_results(response):
        """
        This utility can be used to retrieve results as a list of tuples
        (<result name>, <result resource uri>) to be used in any kind of NER
        related activity.
        :param response: string, the response body
        :return: list of tuples
        """
        # importing JSONDecoder to convert the response body in a dictionary
        from json import JSONDecoder
        decoder = JSONDecoder()

        results = []

        # converting to dictionary
        if response is not None:
            response_dict = decoder.decode(response)
            for res in response_dict["results"]:
                results.append((res["label"], res["uri"]))
            return results
        else:
            return results