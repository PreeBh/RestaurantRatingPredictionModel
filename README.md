# RestaurantRatingPredictionModel


This Use case is a problem set in iNeuron.ai site .
This dataset has been obtained by scraping the TA website for information about restaurants.

The data is a .csv file structured as follow:
o	Name: name of the restaurant
o	City: city location of the restaurant
o	Cuisine Style: cuisine style(s) of the restaurant, in a Python list object (94 046 non-null)
o	Ranking: rank of the restaurant among the total number of restaurants in the city as a float object (115 645 non-null)
o	Rating: rate of the restaurant on a scale from 1 to 5, as a float object (115 658 non-null)(Target Column)
o	Price Range: price range of the restaurant among 3 categories , as a categorical type (77 555 non-null)
o	Number of Reviews: number of reviews that customers have let to the restaurant, as a float object (108 020 non-null)
o	Reviews: 2 reviews that are displayed on the restaurants scrolling page of the city, as a list of list object where the first list contains the 2 reviews, and the second le dates when these reviews were written (115 673 non-null)
o	URL_TA: part of the URL of the detailed restaurant page that comes after 'www.tripadvisor.com' as a string object (124 995 non-null)
o	ID_TA: identification of the restaurant in the TA database constructed a one letter and a number (124 995 non-null)

This model predicts the restaurant ratings based on given attributes 
