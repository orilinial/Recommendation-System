import torch
from torch.utils.data import Dataset


class NetflixPrizeDataset(Dataset):
    """Netflix prize dataset. Taken from https://www.kaggle.com/netflix-inc/netflix-prize-data/home"""

    def __init__(self, ratings_file="./netflix-prize-data/all_data.txt",
                 parsed_file=None,
                 train_movie_ids_mapping=None, dates_range=None, saved_parsed_fname=None, rev_type=None,
                 train_ds=None):
        """
        Args:
            ratings_file (string): Path to the rating file.
        """
        '''
        Variables:
        self.num_of_movies: Number of movies in the dataset.
        self.reviewer_ids_mapping: is a dictionary which maps a reviewer index from the data files to a new 
                                   continuous ordered index (starts from 0)
        self.movie_ids_mapping: is a dictionary which maps a movie index from the data files to a new 
                                continuous ordered index (starts from 0)
        self.reviewer_ratings: For each reviewer index, we have a dictionary with 4 items:
                                reviewer_ratings[reviewer_idx] = {"train_movies": [],
                                                                  "train_ratings": [],
                                                                  "test_movies": [],
                                                                  "test_ratings": []}
                                where each list contains the movies indices and corresponding movies ratings.
        '''
        self.movie_ids_mapping = train_movie_ids_mapping or {}
        self.rev_type = rev_type
        self.train_ds = train_ds
        try:
            if parsed_file is None:
                raise FileNotFoundError
            # Try loading the parsed file
            loaded_data = torch.load(parsed_file)
            self.num_of_movies = loaded_data["num_of_movies"]
            self.reviewer_ids_mapping = loaded_data["reviewer_ids_mapping"]
            self.movie_ids_mapping = loaded_data["movie_ids_mapping"]
            self.reviewer_ratings = loaded_data["reviewer_ratings"]
            print("Loaded " + str(self.reviewer_ratings.__len__()))
        except FileNotFoundError:
            movie_idx = -1
            del_last_movie = False
            last_movie_key = None
            with open(ratings_file) as f:
                i = 0
                self.reviewer_ids_mapping = {}
                self.reviewer_ratings = {}

                # Expecting file in the following format:
                #   movie_index:
                #   reviewer_id,rating,date
                #   reviewer_id,rating,date
                #   reviewer_id,rating,date
                #   ...
                # Read the data file line by line
                skip_movie = False
                for line in f:
                    # Check if line is a movie index (new movie)
                    if skip_movie and line[-2] != ":":
                        continue
                    if line[-2] == ":":
                        skip_movie = False
                        movie_idx_original = int(line[:-2])
                        if train_movie_ids_mapping is not None:
                            if movie_idx_original not in train_movie_ids_mapping.keys():
                                # If train_movie_ids_mapping is not None, we are generating the test set.
                                #   in such case, if the movie does not appear in the train set, we want to skip it
                                skip_movie = True
                                continue
                            if i != 0:
                                self.merge_dicts()
                            movie_idx = train_movie_ids_mapping[movie_idx_original]
                        else:
                            # del_last_movie will be True if the previous movie had no reviews in the dates_range
                            if del_last_movie:
                                self.movie_ids_mapping.pop(last_movie_key, None)
                            elif i != 0:
                                self.merge_dicts()
                            # Get index for the new movie
                            movie_idx = self.movie_ids_mapping.__len__()
                            self.movie_ids_mapping[movie_idx_original] = movie_idx
                            del_last_movie = True
                            last_movie_key = movie_idx_original
                        print("Processing movie " + str(movie_idx_original))

                        # assert(movie_idx + 1 == int(line[:-2]) - 1)
                        # movie_idx = int(line[:-2]) - 1

                        # The following variables are temporary since we don't know if the movie is valid or not
                        # They will be merged to the general "self.reviewer_ids_mapping and self.reviewer_ratings" iff
                        #   the movie is valid (has at least 1 review inside the training dates range)
                        self.reviewer_ratings_of_movie = {}
                        self.reviewer_ids_mapping_of_movie = {}
                        continue

                    # Get review data
                    reviewer_id, rating, date = line.split(",")

                    # Check if review is in date ranges
                    if not self.date_in_range_(date, dates_range):
                        continue
                    del_last_movie = False
                    self.add_review(reviewer_ids_mapping=self.reviewer_ids_mapping_of_movie,
                                    reviewer_ratings=self.reviewer_ratings_of_movie,
                                    reviewer_id=reviewer_id, movie_idx=movie_idx, rating=rating, rev_type=self.rev_type)

                    i += 1
                    if i % 100000 == 0:
                        print("line " + str(i))
            self.num_of_movies = movie_idx + 1
            if saved_parsed_fname is not None:
                torch.save({"num_of_movies": self.num_of_movies,
                            "reviewer_ids_mapping": self.reviewer_ids_mapping,
                            "movie_ids_mapping": self.movie_ids_mapping,
                            "reviewer_ratings": self.reviewer_ratings},
                           saved_parsed_fname)
        if train_movie_ids_mapping is not None:
            self.num_of_movies = len(train_movie_ids_mapping)
        if self.train_ds is not None:
            self.test_reviewer_idx_to_train_reviewer_idx = {}
            # # Create reversed mapping for train to data index
            # train_reviewer_idx_to_data_reviewer_idx = {v: k for k, v in self.train_ds.reviewer_ids_mapping.items()}
            # Create reversed mapping for train to data index
            test_reviewer_idx_to_data_reviewer_idx = {v: k for k, v in self.reviewer_ids_mapping.items()}
            for reviewer_idx in test_reviewer_idx_to_data_reviewer_idx.keys():
                data_reviewer_idx = test_reviewer_idx_to_data_reviewer_idx[reviewer_idx]
                if data_reviewer_idx not in self.train_ds.reviewer_ids_mapping.keys():
                    del self.reviewer_ratings[reviewer_idx]
                else:
                    self.test_reviewer_idx_to_train_reviewer_idx[reviewer_idx] = self.train_ds.reviewer_ids_mapping[data_reviewer_idx]
            new_reviewer_ratings = {}
            new_test_reviewer_idx_to_train_reviewer_idx = {}
            for new_reviewer_idx, reviewer_idx in enumerate(self.reviewer_ratings.keys()):
                new_reviewer_ratings[new_reviewer_idx] = self.reviewer_ratings[reviewer_idx]
                data_reviewer_idx = test_reviewer_idx_to_data_reviewer_idx[reviewer_idx]
                self.reviewer_ids_mapping[data_reviewer_idx] = new_reviewer_idx
                new_test_reviewer_idx_to_train_reviewer_idx[new_reviewer_idx] = self.train_ds.reviewer_ids_mapping[data_reviewer_idx]
            self.test_reviewer_idx_to_train_reviewer_idx = new_test_reviewer_idx_to_train_reviewer_idx
            self.reviewer_ratings = new_reviewer_ratings

            print("Total length after users removal: " + str(len(self.reviewer_ratings)))

    def merge_dicts(self):
        for reviewer_idx in self.reviewer_ids_mapping_of_movie.keys():
            reviewer_new_idx = self.reviewer_ids_mapping_of_movie[reviewer_idx]
            for rev_type in ["train", "test"]:
                movies_lst, rating_lst = (self.reviewer_ratings_of_movie[reviewer_new_idx][rev_type + "_movies"],
                                          self.reviewer_ratings_of_movie[reviewer_new_idx][rev_type + "_ratings"])
                for movie, rating in zip(movies_lst, rating_lst):
                    self.add_review(self.reviewer_ids_mapping, self.reviewer_ratings, reviewer_idx, movie, rating,
                                    rev_type=rev_type)

    @staticmethod
    def add_review(reviewer_ids_mapping, reviewer_ratings, reviewer_id, movie_idx, rating, rev_type):
        # If it's a new reviewer, create reviewer mapping
        if reviewer_id not in reviewer_ids_mapping.keys():
            reviewer_ids_mapping[reviewer_id] = reviewer_ids_mapping.__len__()
        reviewer_new_idx = reviewer_ids_mapping[reviewer_id]
        if reviewer_new_idx not in reviewer_ratings.keys():
            reviewer_ratings[reviewer_new_idx] = {"train_movies": [],
                                                  "train_ratings": [],
                                                  "test_movies": [],
                                                  "test_ratings": []}

        # Add review to reviewer's ratings
        reviewer_ratings[reviewer_new_idx][rev_type + "_movies"].append(int(movie_idx))
        reviewer_ratings[reviewer_new_idx][rev_type + "_ratings"].append(int(rating))

    @staticmethod
    def date_in_range_(date, dates_range):
        if dates_range is not None and dates_range["start"] is not None:
            assert (dates_range["end"] is not None)
            year, month, _ = date[:-1].split("-")
            year = int(year)
            month = int(month)
            if not (dates_range["start"]["year"] <= year <= dates_range["end"]["year"]):
                return False
            if dates_range["start"]["year"] == year and not (dates_range["start"]["month"] <= month):
                return False
            if dates_range["end"]["year"] == year and not (month <= dates_range["end"]["month"]):
                return False
        return True

    def __len__(self):
        return len(self.reviewer_ratings)

    def __getitem__(self, idx):
        vec = torch.zeros(self.num_of_movies)
        vec[self.reviewer_ratings[idx][self.rev_type + "_movies"]] = \
            torch.FloatTensor(self.reviewer_ratings[idx][self.rev_type + "_ratings"])
        if self.train_ds is None:
            return vec, vec
        else:
            return self.train_ds[self.test_reviewer_idx_to_train_reviewer_idx[idx]][0], vec


def _netflix_by_dates(fname, train_range, test_range):
    ratings_file = "./netflix-prize-data/all_data.txt"
    dataset_train = NetflixPrizeDataset(ratings_file=ratings_file, dates_range=train_range,
                                        parsed_file="train_" + fname + ".pth",
                                        saved_parsed_fname="train_" + fname + ".pth", rev_type="train")
    dataset_test = NetflixPrizeDataset(ratings_file=ratings_file, dates_range=test_range,
                                       train_movie_ids_mapping=dataset_train.movie_ids_mapping,
                                       parsed_file="test_" + fname + ".pth",
                                       saved_parsed_fname="test_" + fname + ".pth", rev_type="test",
                                       train_ds=dataset_train)
    return dataset_train, dataset_test


def netflix_full():
    train_range = {"start": {"year": 1999, "month": 12},
                   "end": {"year": 2005, "month": 11}}
    test_range = {"start": {"year": 2005, "month": 12},
                  "end": {"year": 2005, "month": 12}}
    return _netflix_by_dates("full", train_range, test_range)


def netflix_3months():
    train_range = {"start": {"year": 2005, "month": 9},
                   "end": {"year": 2005, "month": 11}}
    test_range = {"start": {"year": 2005, "month": 12},
                  "end": {"year": 2005, "month": 12}}
    return _netflix_by_dates("3months", train_range, test_range)


def netflix_6months():
    train_range = {"start": {"year": 2005, "month": 6},
                   "end": {"year": 2005, "month": 11}}
    test_range = {"start": {"year": 2005, "month": 12},
                  "end": {"year": 2005, "month": 12}}
    return _netflix_by_dates("6months", train_range, test_range)


def netflix_1year():
    train_range = {"start": {"year": 2004, "month": 6},
                   "end": {"year": 2005, "month": 5}}
    test_range = {"start": {"year": 2005, "month": 6},
                  "end": {"year": 2005, "month": 6}}
    return _netflix_by_dates("1year", train_range, test_range)
