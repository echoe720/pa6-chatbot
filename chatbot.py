# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
from collections import defaultdict
import numpy as np
import regex as re 
from porter_stemmer import PorterStemmer
from itertools import combinations


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'ELON'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)

        self.user_ratings = np.zeros((len(ratings)))

        self.input_count = 0
        self.reachedRecommendation = False
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################
        if self.creative:
            greeting_message = "Hello, I'm ELON! I'm going to help you find a movie to watch. Tell me about a movie you watched and whether you liked the movie."
        else:
            greeting_message = "Hello, I'm ELON! I'm going to help you find a movie to watch. Tell me about a movie you watched and whether you liked the movie. Please put the name of the movie in quotation marks."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "I hope you have a great day and remember to full send!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################
    def tokenize(self, raw_text):
        """
        Tokenizes raw_text by seperating the input string by puncuation. Converts contractions
        into full words. Returns a list of tokens
        """
        tokens = re.findall(r"\w+", raw_text)
        for x in range(len(tokens)):
            if tokens[x] == 's':
                tokens[x] = 'is'
            elif tokens[x] == 't':
                tokens[x] = 'not'
            elif tokens[x] == 're':
                tokens[x] = 'are'
        return tokens

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        # self.creative = True
        if self.creative:
            movieTitles = self.extract_titles(self.preprocess(line)) #list of possible MovieTitles  #should call find_movies_closest_to_title in here? Something to think about-- I think the answer is no for now. 
            print('movieTitles: ', movieTitles)
            
            affirmative = ["yes", "sure", "ok", "yeah", "y", "affirmative", "i guess so", "fine", "always"]
            negative = ["no", "nah", "never", "negative", "n", "no thanks", "no, thanks", "nope"]
            movieIdx = 0

            if len(movieTitles) == 0: # added this check for no movie, 
                #what about the case where mentioned movie is not in the database??
                # process for emotion

                #what is your name


                # if len(movieIdx) == 0: #if no movie was initially found 

                i = 0
                if "\"" in line: #weak logic check here.. come back and fix
                    allPossibleMovies = []
                    allMovies = re.findall('"([^"]*)"', line) #explore all possible substrings in the given input sentence??  YES-- like extract_titles
                    for movie in allMovies:
                        allPossibleMovies.extend(self.find_movies_closest_to_title(movie)) # the input should be the "incorrect" movie title
                    lenMovies = len(allPossibleMovies)
                    # are there still no possible movies?
                    if lenMovies == 0:
                        return "Huh, I\'m not sure what movie you are talking about. What's a different movie that you have seen?"
                    else:
                        # multiple movies, so we disambiguate
                        if lenMovies >= 2:   
                            unclear = True
                            while unclear:
                                unclear_movie_titles = []
                                for id in allPossibleMovies:
                                    unclear_movie_titles.append(self.titles[id][0])
                                answer = input("I found more with a title similar to " + movieTitle[0] + f". Which of these is the one you are telling me about: {str(unclear_movie_titles)}?\n> ")
                                if answer in unclear_movie_titles:
                                    return "Great, you liked \"" + str(answer) + "\"."
            
                                movieIndices = self.disambiguate(answer, movieIdx)
                                if len(movieIndices) == 1: unclear = False
                                #is break statement missing?
                       
                        else:  # only one possible movie so we pick that one and ask if that was the right one
                            answer = input("Did you mean " + self.titles[allPossibleMovies[0]][0] + "?")
                            movieIdx = allPossibleMovies[0]
                            if answer in negative:
                                return "Huh, I\'m not sure what movie you are talking about. What's a different movie that you have seen?"
                        self.input_count += 1
                        if self.input_count == 5:
                            return self.giveRecommendations(affirmative, negative)
                        else:
                            return "Great, you liked \"" + self.titles[movieIdx][0] + "\"."

                # foreign_title = self.find_foreign(line)
                # if foreign_title != "":
                #     self.input_count += 1
                #     if sentiment == 1:
                #             return "Ok, you liked \"" + foreign_title[0] + "\"! Tell me what you thought of another movie."
                #     elif sentiment == -1:
                #         return "Ok, you didn't like \"" + foreign_title[0] + "\"! Tell me what you thought of another movie."
                #     else: 
                #         self.input_count -= 1
                #         return "I'm confused, did you like \"" + foreign_title[0] + "\"? Please try to clarify if you liked the movie."  
                
                question_keywords = ["can you", "what is"]
                line = line.lower()
                for key in question_keywords:
                    if key in line:
                        idx = line.index(key)
                        line = line[idx + len(key):]
                        if "me" in line or "my" in line:
                            line = re.sub("(me)", "you", line)
                            line = re.sub("(my)", "your", line)
                        elif "you" in line or "your" in line:
                            line = re.sub("(your)", "my", line)
                            line = re.sub("(you)", "me", line)
                            
                        if key == "can you":
                            return "Sorry, I can't" + line + "."
                        elif key == "what is":
                            return "I don't know what" + line + " is."

                feeling_keywords = ["i am feeling", "i am", "i'm", "feeling", "i feel", "you are", "you're"]
                for key in feeling_keywords:
                    if key in line:
                        idx = line.index(key)
                        line = line[idx + len(key):]
                        line_sentiment = self.extract_sentiment(line)
                        if line_sentiment == -1:
                            return "Sorry you feel" + line + ". ELON is very sorry."
                        elif line_sentiment == 1:
                            return "Glad you feel" + line + ". ELON is very happy :)"
                if "\"" in line:
                    return "I wasn't able to find that movie."
                else:
                    return "You don't seem to be talking about movies."
            else: #if len(movieTitles) != 0 --> if there is one or more possible movies that the user was indicating (e.g. input: Harry Potter --> the 6 harry potter movies)
                movieIndices = self.find_movies_by_title(movieTitles[0]) # THIS LOGIC IS INCORRECT-- COME BACK!!  this case doesn't work: I love Fists in the Pocket (Pugni in tasca, I) (1965)
                print('movieIndices: ', movieIndices) #empty in cases of certain movies right now.....
                movieIdx = movieIndices[0] #initializing

                # multiple movies, so we disambiguate
                if len(movieIndices) >= 2:
                    unclear = True
                    # print(movieTitle)
                    
                    while unclear:
                        unclear_movie_titles = []
                        for id in movieIndices:
                            unclear_movie_titles.append(self.titles[id][0])
                        answer = input("I found more than one movie called " + str(movieTitles[0]) + f". Which of these is the one you are telling me about: {str(unclear_movie_titles)}?\n> ")
                        if answer in unclear_movie_titles:
                            return "Great, you liked \"" + str(answer) + "\"."
                        movieIdx = self.disambiguate(answer, movieIndices)
                        if len(movieIdx) == 1: 
                            unclear = False

                sentiment = self.extract_sentiment_creative(line)
                # print('Sentiment: ', sentiment)

                if self.user_ratings[movieIdx] == 0 and sentiment != 0:
                    self.input_count += 1
                self.user_ratings[movieIdx] = sentiment

                if self.input_count == 5:
                    response = self.giveRecommendations(affirmative, negative)
                else: #if self.input_count < 5
                    if sentiment >= 1:
                        return "Ok, you liked \"" + movieTitles[0] + "\"! Tell me what you thought of another movie."
                    elif sentiment <= -1:
                        return "Ok, you didn't like \"" + movieTitles[0] + "\"! Tell me what you thought of another movie."
                    else: 
                        self.input_count -= 1
                        return "I'm confused, did you like \"" + movieTitles[0] + "\"? Please try to clarify if you liked the movie."
            response = self.goodbye()




        else: #starter mode case(non-creative)
            # In starter mode, your chatbot will help the user by giving movie recommendations. 
            # It will ask the user to say something about movies they have liked, and it will come up with a recommendation based on those data points. 
            # The user of the chatbot in starter mode will follow its instructions, so you will deal with fairly restricted input. 
            # Movie names will come in quotation marks and expressions of sentiment will be simple.
            movieTitle = self.extract_titles(self.preprocess(line))
            sentiment = self.extract_sentiment(line)
            # print(sentiment)

            if len(movieTitle) == 0: # added this check for no movie, 
                #what about the case where mentioned movie is not in the database??
                response = "I'm not sure what movie you are talking about. Make sure you are puttting quotation marks around the movie name so that I can tell what movie you are talking about."
                return response
            elif len(movieTitle) >= 2: 
                response = "Please tell me about one movie at a time. What's one movie you have seen?"
                return response
            else:
                # print(movieTitle[0])
                movieIndices = self.find_movies_by_title(movieTitle[0]) # added [0] here
                if len(movieIndices) == 0:
                    return "Huh, I\'m not sure what movie you are talking about. What's a different movie that you have seen?"
                elif len(movieIndices) >= 2:
                    return "I found more than one movie called " + movieTitle[0] + ". Which one were you trying to tell me about?"
                else:
                    movieIdx = movieIndices[0]
                    # print(self.titles[movieIdx][0])
                    
                    if self.user_ratings[movieIdx] == 0 and sentiment != 0: #if the user hasn't watched the movie already and the sentiment of the current movie is positive or negative:
                        self.input_count += 1
                    self.user_ratings[movieIdx] = sentiment

                    if self.input_count == 5:
                        self.user_ratings = self.user_ratings
                        self.ratings = self.binarize(self.ratings)
                        # print(len(self.titles)) #9125
                        # print(self.user_ratings.shape) #(9125, )
                        # print(self.ratings.shape) #(9125, 671)
                        
                        #get number of movies that the user haven't watched, and pass it in as k to recommend
                        k = len([rating for rating in self.user_ratings if rating == 0])
                        # print(k) #9120
                        
                        recommendations = self.recommend(self.user_ratings, self.ratings, k, creative=self.creative) #prior k: len(self.titles) - 1.  at most, k will be number of movies
                        # print(recommendations) #currently giving Women of 69 unboxed followed by Gay Desperado for all inputs as first two recs"

                        i = -1
                        affirmative = ["yes", "sure", "ok", "yeah", "y", "affirmative", "i guess so", "fine", "always"]
                        negative = ["no", "nah", "never", "negative", "n", "no thanks", "no, thanks", "nope"]
                        answer = "yes" 
                        while (answer in affirmative):
                            i += 1
                            # print(recommendations)
                            # print(recommendations[i][0])
                            answer = input('I think you\'ll enjoy watching \"' + self.titles[recommendations[i]][0] + '\"! Would you like another recommendations?\n').lower()
                            if answer in negative:
                                # response = "Have a nice day. Fullsend!"
                                break
                            elif answer not in affirmative and answer not in negative:
                                currInput = input("Please input \"Yes\" or \"No\". ELON is disappointed in you. Let's try again. Would you like more recommendations?\n").lower()
                                while (currInput != 'yes' and currInput != 'no'):
                                    currInput = input("Please input \"Yes\" or \"No\". ELON is disappointed in you. Let's try again. Would you like more recommendations?\n")
                                answer = currInput
                                
                            if i == len(self.titles):
                                response = "We have no more recommendations -- Have a nice day. Fullsend!"
                    else: #if self.input_count < 5
                        if sentiment == 1:
                            return "Ok, you liked \"" + self.titles[movieIdx][0] + "\"! Tell me what you thought of another movie."
                        elif sentiment == -1:
                            return "Ok, you didn't like \"" + self.titles[movieIdx][0] + "\"! Tell me what you thought of another movie."
                        else: 
                            self.input_count -= 1
                            return "I'm confused, did you like \"" + movieTitle[0] + "\"? Please try to clarify if you liked the movie."
            response = self.goodbye()
        return response


    def giveRecommendations(self, affirmative, negative):
        self.ratings = self.binarize(self.ratings)
        
        #get number of movies that the user haven't watched, and pass it in as k to recommend?
        k = len([rating for rating in self.user_ratings if rating == 0])
        recommendations = self.recommend(self.user_ratings, self.ratings, k, creative=self.creative) #prior k: len(self.titles) - 1.  at most, k will be number of movies
        i = -1
        answer = "yes" 
        response = ""
        while (answer in affirmative):
            i += 1
            if i == len(self.titles):
                response = "We have no more recommendations -- Have a nice day. Fullsend!"
            # print(recommendations)
            # print(recommendations[i])
            # print(recommendations[i][0])
            answer = input('I think you\'ll enjoy watching\"' + self.titles[recommendations[i]][0] + '\"! Would you like another recommendations?\n').lower()
            if answer in negative:
                response = "Have a nice day. Fullsend!"
                break
            elif answer not in affirmative and answer not in negative:
                currInput = input("Please input \"Yes\" or \"No\". ELON is disappointed in you. Let's try again. Would you like more recommendations?\n").lower()
                while (currInput != 'yes' and currInput != 'no'):
                    currInput = input("Please input \"Yes\" or \"No\". ELON is disappointed in you. Let's try again. Would you like more recommendations?\n")
                answer = currInput
                
            
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.
        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """ 
        if not self.creative:
            return re.findall('"([^"]*)"', preprocessed_input)
        else:
            movie_titles = []
            quote = re.findall('"([^"]*)"', preprocessed_input)
            if quote != []:
                foreign = self.find_foreign(quote)
                if foreign != []:
                    return foreign # just added to handle foreign
           
            for title in self.titles:
                movie_titles.append(sorted(self.tokenize(title[0].lower()))) # now storing movie titles tokenized list, sorted so out of order matches
                movie_titles.append(sorted(self.tokenize(re.sub("( \([0-9]+\))", "", title[0]).lower()))) # add both the title w/ year and without to list
            
            split_words = re.sub(r'[^\w\s()]', '', preprocessed_input.lower()).split()

            indices = {}
            index_list = []
            result = []
            for start in range(len(split_words)):
                for end in range(start, len(split_words)):
                    substring = split_words[start:end+1]
                    processed_string = ' '.join(substring)
                    if sorted(self.tokenize(processed_string)) in movie_titles:
                        if start in indices:
                            if indices[start] < end:
                                indices[start] = end
                        else:
                            indices[start] = end
            for key in indices:
                index_list.append((key, indices[key]))
            index_list = sorted(index_list)

            # eliminate overlapping indices (eliminate the shorter one)
            i = 1
            while i < len(index_list):
                # print(index_list)
                # print(index_list[i][0])
                # print(index_list[i-1][1])
                if index_list[i][0] <= index_list[i-1][1]: # there is overlap
                    len1 = index_list[i][1] - index_list[i][0] + 1
                    len2 = index_list[i-1][1] - index_list[i-1][0] + 1
                    if len2 > len1:
                        index_list.pop(i) # pop the smaller one
                        i -= 1
                    else:
                        index_list.pop(i-1)
                        i -= 1
                i += 1 
            
            for index in index_list:
                movie_title = ' '.join(split_words[index[0]:index[1] + 1])
                result.append(movie_title)
            return result

    # assumptions: foreign titles don't have years, only aka and a.k.a. <== get these checked
    # edge case: 792%Yes, Madam (a.k.a. Police Assassins) (a.k.a. In the Line of Duty 2) (Huang gu shi jie) (1985)%Action
    # edge 7603%Babies (Bébé(s)) (2010)%Documentary
    # working 5190%Soldier of Orange (a.k.a. Survival Run) (Soldaat van Oranje) (1977)%Drama|Thriller|War

    def find_foreign(self, titleList):
        res = []
        foreign_dict = {}
        title = str(titleList[0])
        titleYear = re.findall("(\([0-9]+\))", title) 
        title = re.sub( "(\([0-9]+\))", "", title) # filter out year from original title
        titleWords = self.tokenize(title)
        foreign_titles = []
        foreign_titles_set = {}
        # tokenize movie title, mamke sure every token in movie title input also in tokenized representation of movie
        for i in range(len(self.titles)):
            currTitle = self.titles[i][0]
            # normalTitle = currTitle
            currTitle = re.sub("(\([0-9]+\))", "", currTitle)
            currTitle = re.findall("\(.*?\)", currTitle)
            if currTitle != []:
                for subTitle in currTitle:
                    subTitle = re.sub("(a.k.a. )", "", subTitle)
                    subTitle = re.sub("[()]", "", subTitle)
                    foreign_dict[subTitle] = i #normalTitle.strip()
                    foreign_titles.append(subTitle)
                    foreign_titles_set[frozenset(self.tokenize(subTitle))] = i
        # print(foreign_dict)
        # print(foreign_titles)

        if title in foreign_dict:
            print(title)
            return self.titles[foreign_dict[title]]
        else:
            currWords = frozenset(self.tokenize(title))
            if currWords in foreign_titles_set:
                return self.titles[foreign_titles_set[currWords]]
        return res

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        if self.creative:
            movie_titles = []
            res = []
            titleYear = []
            titleWords = []
            for t in self.titles:
                movie_titles.append((t[0].lower())) # now storing movie titles tokenized list, sorted so out of order matches
            title = str(title).lower()
            titleYear.append(re.findall("(\([0-9]+\))", title))
            title = re.sub( "(\([0-9]+\))", "", title) # filter out year from original title
            titleWords = self.tokenize(title)

            if titleWords[0] in ("the", "a", "an", "le", "el", "la"):
                the = titleWords[0]
                titleWords.remove(the)
                newTitle = ' '.join(titleWords)
                newTitle += ", " + the
                title = newTitle
            for i in range(len(movie_titles)):
                currTitle = movie_titles[i]
                currYear = re.findall("(\([0-9]+\))", currTitle)
                if len(titleWords) == 1 and list(titleWords)[0] in self.tokenize(currTitle):
                    if titleYear[0] == [] or titleYear[0] == currYear[0]:
                        res.append(i)
                elif len(titleWords) != 1 and title in currTitle:
                    res.append(i)
                else:
                    currYear = re.findall("(\([0-9]+\))", currTitle)
                    currTitle = re.sub("(\([0-9]+\))", "", currTitle)
                    currWords = self.tokenize(currTitle)
                    sameMovie = True
                    currWords = set(currWords)
                    titleWords = set(titleWords)
                    if ',' in currWords:
                        currWords.remove(',')
                    if ',' in titleWords:
                        titleWords.remove(',')
                    if currWords == titleWords: # the vectorized words are subsets of each other
                        if titleYear == [] or currYear[0] == titleYear[0]: # if there is a year specified in title make sure it matches
                            res.append(i) 
            # print(len(res))
            # if len(res) == 0:
            #     return self.find_foreign(title)
            print(res)
            return res
        else:
            res = []
            titleYear = re.findall("(\([0-9]+\))", title) 
            title = re.sub( "(\([0-9]+\))", "", title) # filter out year from original title
            titleWords = self.tokenize(title)
            # tokenize movie title, mamke sure every token in movie title input also in tokenized representation of movie
            for i in range(len(self.titles)):
                currTitle = self.titles[i][0]
                if currTitle == title:
                    res.append(i)
                else:
                    currYear = re.findall("(\([0-9]+\))", currTitle)
                    currTitle = re.sub("(\([0-9]+\))", "", currTitle)
                    currWords = self.tokenize(currTitle)
                    sameMovie = True
                    currWords = set(currWords)
                    titleWords = set(titleWords)
                    if ',' in currWords:
                        currWords.remove(',')
                    if ',' in titleWords:
                        titleWords.remove(',')
                    if currWords == titleWords: # the vectorized words are subsets of each other
                        if titleYear == [] or currYear[0] == titleYear[0]: # if there is a year specified in title make sure it matches
                            res.append(i) 
            return res

    def extract_regex(self, s):
        if s == "loved":
            return "love"
        if s == "enjoyed":
            return "enjoy"
        s = re.sub( "(d)$", "", s) 
        s = re.sub( "(s)$", "", s) 
        s = re.sub( "(ing)$", "", s) 
        return s

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        pos_count = 0.0
        neg_count = 0.0
        lmd = 1.0
        neg_words = ["never", "not"]

        negated_flag = False
        distance_from_negation = 0
        max_negation_distance = 5
        preprocessed_input = re.sub('"([^"]*)"', "", preprocessed_input).lower()
        preprocessed_input = self.tokenize(preprocessed_input)
        for item in preprocessed_input:
            # print(item)
            if distance_from_negation == max_negation_distance:
                distance_from_negation = 0
                negated_flag = False
            if item not in self.sentiment:
                item = self.extract_regex(item)
            # print(item)
            if item != None and item in neg_words:
                negated_flag = not negated_flag
                continue
            if item not in self.sentiment:
                continue
            elif self.sentiment[item] != None and "pos" in self.sentiment[item]:
                if negated_flag:
                    neg_count += 1.0
                else:
                    pos_count += 1.0
            elif self.sentiment[item] != None and "neg" in self.sentiment[item]:
                if negated_flag:
                    pos_count += 1.0
                else:
                    neg_count += 1.0
            if negated_flag:
                distance_from_negation += 1
        if neg_count == 0.0 and pos_count == 0.0:
            return 0
        elif neg_count ==0:
            return 1
        if pos_count / neg_count > lmd:
            return 1
        elif pos_count / neg_count < lmd:
            return -1
        return 0

    def extract_sentiment_creative(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.
        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.
        
        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        pos_count = 0.0
        neg_count = 0.0
        lmd = 1.0
        neg_words = ["never", "not"]

        negated_flag = False
        distance_from_negation = 0
        max_negation_distance = 5

        # Cases to consider:
        # Case 1. Extreme words behind pos/neg word(s). 
        # Case 2. Words themselves are really strongly positive or negative 
        
        extreme_flag = False
        
        currTitles = self.extract_titles(preprocessed_input) #lowercased title extracted, and currTitles at this point should be a list of size 1 (one title)
        # print('currTitles in sentiment_creative: ', currTitles)

        preprocessed_input = preprocessed_input.lower()
        preprocessed_input = preprocessed_input.replace(currTitles[0], "")
        # preprocessed_input = re.sub(currTitles[0], "", preprocessed_input)
        # print('preprocessed_input in sentiment_creative: ', preprocessed_input)

        # preprocessed_input = re.sub('"([^"]*)"', "", preprocessed_input).lower() #this doesn't work in cases where the input doesn't have "quoted" movies

        extreme_words = re.findall("(:?r+e+a+l+l+y+)|(:?e+x+t+r+e+m+e+l+y+)|(:?v+e+r+y+)", preprocessed_input) #it is okay to just find as many as possible manually, but what about "I don't really like??"
        preprocessed_input = self.tokenize(preprocessed_input)

        extreme_negative_words = open('data/negative.txt', 'r').readlines() #not quite negative-- replace with new txt file. ALL LOWERCASED #just cite it!
        extreme_positive_words = open('data/positive.txt', 'r').readlines() #not quite positive-- replace with new txt file. ALL LOWERCASED
        
        for item in preprocessed_input:
            if item in extreme_words:
                extreme_flag = True         
            if distance_from_negation == max_negation_distance:
                distance_from_negation = 0
                negated_flag = False
            if item not in self.sentiment:
                item = self.extract_regex(item)
            if item != None and item in neg_words:
                negated_flag = not negated_flag
                continue
            if item not in self.sentiment:
                continue
            elif self.sentiment[item] != None and "pos" in self.sentiment[item]:
                if negated_flag: #e.g. "not good"
                    neg_count += 1.0
                    if extreme_flag: #e.g. "not really awesome" and "really not awesome"
                        extreme_flag = False

                else:
                    pos_count += 1.0
                    isExtremePos = False
                    if item in extreme_positive_words:
                        isExtremePos = True

                    if extreme_flag or isExtremePos: #e.g. "REALLY good or Amazing"
                        pos_count += 1.0
                        extreme_flag = False
            elif self.sentiment[item] != None and "neg" in self.sentiment[item]:
                if negated_flag: #e.g. not bad
                    pos_count += 1.0
                    if extreme_flag:  #e.g. not really bad
                        extreme_flag = False
                else:
                    neg_count += 1.0
                    isExtremeNeg = False
                    if item in extreme_negative_words:
                        isExtremeNeg = True

                    if extreme_flag or isExtremeNeg: #e.g. verrry bad / HORRIBLE
                        neg_count += 1.0
                        extreme_flag = False
            if negated_flag:
                distance_from_negation += 1
        if neg_count == 0.0 and pos_count == 0.0:
            return 0
        elif neg_count ==0:
            return 1
        if pos_count / neg_count >= lmd * 2:
            return 2
        elif pos_count / neg_count > lmd:
            return 1
        elif pos_count / neg_count <= lmd / 2:
            return -2
        elif pos_count / neg_count < lmd:
            return -1
        return 0


    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        pass

    def levenshtein(self, s, t): # https://python-course.eu/levenshtein_distance.php
        rows = len(s)+1
        cols = len(t)+1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        for i in range(1, rows):
            dist[i][0] = i
        for i in range(1, cols):
            dist[0][i] = i     
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row-1] == t[col-1]:
                    cost = 0
                else:
                    cost = 2
                dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                    dist[row][col-1] + 1,      # insertion
                                    dist[row-1][col-1] + cost) # substitution
        # for r in range(rows):
        #     print(dist[r])
        return dist[rows-1][cols-1]

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        edit_dist = float('inf')
        title_dist_dict = {}
        title = title.lower()
        for i in range(len(self.titles)):
            iter_title = self.titles[i][0].lower()
            iter_name = re.sub("(\([0-9]+\))", "", iter_title).strip() # compare against stripped movie with no year
            curr_dist = self.levenshtein(iter_name, title)
            if curr_dist <= edit_dist and curr_dist <= max_distance:
                edit_dist = curr_dist
                if curr_dist in title_dist_dict:
                    title_dist_dict[curr_dist].append(i)
                else:
                    title_dist_dict[curr_dist] = [i]
        if edit_dist == float('inf'):
            return []
        min_list = title_dist_dict[edit_dist] # dict contains old min edit distances as well, inefficient 
        result = []
        if min_list is not None:
            for i in min_list:
                result.append(i)
        return result

    def check_substring_clarification(self, clarification, candidates):
        """
        Looks through the clarification provided by the user for substrings of
        two or more words that match a substring in a candidate movie title.

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        new_candidates = set()
        words = self.tokenize(clarification.lower())
        for indice in candidates:
            for start_loc in range(len(words)+1):
                for end_loc in range(start_loc + 2, len(words)+1):
                    phrase = ""
                    for location in range(start_loc, end_loc):
                        phrase += f" {words[location]}"
                    phrase = phrase[1:]
                    if phrase in self.titles[int(indice)][0].lower():
                        new_candidates.add(int(indice))
        return list(new_candidates)
    
    def sort_movies_by_date(self, input_list):
        """sorts an input_list of movie indices by the date in the title of the movie.
        The order is oldest to newest. Returns the sorted list
        """
        movie_list = []
        for movie in input_list:
            year = int(re.findall("\(([0-9]+)\)", self.titles[int(movie)][0])[0])
            movie_list.append((year, movie))
        movie_list.sort()
        result = []
        for year, movie in movie_list:
            result.append(movie)
        return result

    def clarify_oldest_newest(self, clarification, candidates, full=False):
        """
        Looks through the clarification provided by the user for phrases like 
        "the newer movie", or "the original version" and returns list of indices
        corresponding to movies that fit that clarification.

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :param full: use numbers in the clarification
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        new_candidates = set()
        # dictionary of phrases to look for
        order_phrases = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "seventh": 6, "eight": 7, "ninth": 8, "tenth": 9, "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4, "6th": 5, "7th": 6, "8th": 7, "9th": 8, "10th": 9}
        full_phrases = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "8": 8, "10": 9}
        latest = ["latest", "recent", "new", "newest", "newer", "later"]
        earliest = ["earliest", "old", "oldest", "older", "earlier", "original"]
        words = self.tokenize(clarification.lower())
        oldest = False
        newest = False
        # checks the clarification input
        for word in earliest:
            if word in words:
                oldest = True
        for word in latest:
            if word in words:
                newest = True
        # finds the oldest movie of the candidates
        phrases = {}
        if not full:
            phrases = order_phrases
        else:
            phrases = full_phrases
        if oldest:
            offset = 0
            for order_phrase in phrases:
                if order_phrase in clarification:
                    offset = phrases[order_phrase]
            return [self.sort_movies_by_date(candidates)[offset]]
        # finds the newest movie of the candidates
        if newest:
            offset = 0
            for order_phrase in phrases:
                if order_phrase in clarification:
                    offset = phrases[order_phrase]
            movie_list = self.sort_movies_by_date(candidates)
            return [movie_list[len(movie_list) - offset - 1]]
    
    def clarify_by_order(self, clarification, candidates, full=False):
        """
        Looks through the clarification provided by the user for phrases like 
        "the second one, or "the last on the list" and returns list of indices
        corresponding to movies that fit that clarification.

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :param full: use numbers in the clarification
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        clarification = clarification.lower()
        order_phrases = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "seventh": 6, "eight": 7, "ninth": 8, "tenth": 9, "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4, "6th": 5, "7th": 6, "8th": 7, "9th": 8, "10th": 9}
        reverse_phrases = ["from the end", "from the back", "to last", "to the last", "from last", "from the last", "to the end", "to the back"]
        full_phrases = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "8": 8, "10": 9}
        phrases = {}
        if not full:
            phrases = order_phrases
        else:
            phrases = full_phrases
        for order_phrase in phrases:
            for reverse_phrase in reverse_phrases:
                if f"{order_phrase} {reverse_phrase}" in clarification:
                    return [candidates[len(candidates) - phrases[order_phrase] - 1]]
            if order_phrase in clarification:
                return [candidates[phrases[order_phrase]]]
        if "last" in clarification or "latter" in clarification:
            return [candidates[len(candidates) - 1]]
        return []

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        new_candidates = set()
        # checks for phrases like "the original" or "the latest one"
        results = self.clarify_oldest_newest(clarification, candidates)
        if results:
            for x in results:
                new_candidates.add(x)
        # checks for a location on the list of candidates
        results = self.clarify_by_order(clarification, candidates)
        if results and not new_candidates:
            for x in results:
                new_candidates.add(x)
        # checks if substrings of two or more clairifcation words are in the title
        results = self.check_substring_clarification(clarification, candidates)
        if results:
            for x in results:
                new_candidates.add(x)
        
        # finds the movie or movie that matches the name or year in the clarification
        # extract info from the clarification
        titleYearParen = re.findall("(\([0-9]+\))", clarification)
        titlenum = re.findall("([0-9]+)", clarification)
        titleYear = re.findall("([0-9]{2,4})", clarification)
        for indice in candidates:
            # check to see if the clarification info is a match for the candidate info
            candidate_year = re.findall("(\([0-9]+\))", self.titles[int(indice)][0])[0]
            candidate_name = re.sub( "(\([0-9]+\))", "", self.titles[int(indice)][0])
            if clarification in candidate_name:
                new_candidates.add(int(indice))
                # new_candidates.add(1) matches "Titanic 2"
            elif titleYearParen != "" and titleYearParen == candidate_year: # matches  "Titanic (1973)"
                new_candidates.add(int(indice))
            elif clarification == candidate_year: # matches"(1973)"
                new_candidates.add(int(indice))
            elif clarification == candidate_year[1:5]: # matches "1973"
                new_candidates.add(int(indice))
            elif clarification == candidate_year[3:5]: # matches "73"
                new_candidates.add(int(indice))
            elif titlenum == candidate_year[1:5]: # matches "Titanic 1973"
                new_candidates.add(int(indice))
            elif titleYear != [] and titleYear[0] in self.titles[int(indice)][0]:
                new_candidates.add(int(indice))
        new_candidates = list(new_candidates)
        # if there are no movies that match, try treating numbers like "3" as positional indicators        
        if new_candidates == []:
            round_two_candidates = set()
            # checks for phrases like "the original" or "the latest one"
            results = self.clarify_oldest_newest(clarification, candidates, True)
            if results:
                for x in results:
                    round_two_candidates.add(x)
            # checks for a location on the list of candidates
            results = self.clarify_by_order(clarification, candidates, True)
            if results and not round_two_candidates:
                for x in results:
                    round_two_candidates.add(x)
            if list(round_two_candidates) == []:
                return candidates
            else:
                return list(round_two_candidates)
        return new_candidates

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        
        ratingsCopy = ratings.copy()
        binarized_ratings = np.where(ratings <= threshold, ratings, 1)
        binarized_ratings = np.where(ratings > threshold, binarized_ratings, -1)
        ratingsCopy = np.where(ratingsCopy != 0, binarized_ratings, 0)

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return ratingsCopy

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u,v)
        normalizing_factor = np.linalg.norm(u) * np.linalg.norm(v)
        if normalizing_factor == 0:
            similarity = 0
        else:
            similarity = np.divide(similarity, normalizing_factor)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.


        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        movieScore = {}
        rowLen = ratings_matrix.shape[0]
        # print(user_ratings)
        for i in range(rowLen):
            movieScore[i] = 0
            for j in range(rowLen):
                if i != j and user_ratings[j] != 0:
                    movieScore[i] += self.similarity(ratings_matrix[i], ratings_matrix[j]) * user_ratings[j]

        scoresToMovie = []
        for movie in movieScore:
            if user_ratings[movie] == 0: #if the user has not watched the movie 
                scoresToMovie.append((movieScore[movie], movie)) #do the movie indices here correspond with the "acutal" movie indices in self.titles?
            
        scoresToMovie = sorted(scoresToMovie, reverse=True) #sorted(ratingsToMovie, key=lambda rating : rating[0], reverse=True)
        # print(len(scoresToMovie))
        # print(k)
        for idx in range(k):
            recommendations.append(scoresToMovie[idx][1])
        
        
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """

        # test for find_movies_by_title
        debug_info = 'debug info'
        # id1 = "An American in Paris (1951)"
        # id2 = "The Notebook (1220)"
        # id3 = "Titanic"
        # id4 = "Scream"
        # l = list([id1])
        # for elem in l:
        #     print(elem, self.find_movies_by_title(elem))

        # test for extract_sentiment
        # l2 = ["I didn't really like \"Titanic (1997)\"", "I never liked \"Titanic (1997)\"", "I really enjoyed \"Titanic (1997)\"",
        #         "I saw \"Titanic (1997)\".",  "\"Titanic (1997)\" started out terrible, but the ending was totally great and I loved it!",
        #         "I loved \"10 Things I Hate About You\""]
        # for elem in l2:
        #     print(elem, self.extract_sentiment(elem))
        
        id1 = "I liked The NoTeBoOk (2004)!"
        id2 = "I thought 10 things i hate about you was great"
        id3 = "I liked The Notebook and I liked 10 things i hate about you was great"
        id4 = "Se7en"
        id5 = "La Guerre du feu"
        id6 = "10 things i HATE about you"
        id7 = "Titanic"
        id8 = "Scream"
        l3 = list([id1, id2, id3, id4, id5, id6, id7, id8])
        for elem in l3:
            # print(elem)
            extracted_titles = self.extract_titles(elem)
            # print(extracted_titles)
            if len(extracted_titles) == 0:
                print(elem, self.find_movies_by_title(elem))
            else:
                ans = []
                for title in self.extract_titles(elem):
                    print(elem, self.find_movies_by_title(title))
            
        # print(self.find_foreign(id5))
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return "Hi! This is a temporary intro string."
        """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')

