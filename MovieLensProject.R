############################################################
# Movielens Recommendation System (Capstone Project)
# By Richard Jonyo
# Purpose: Calculate RMSE to be used to evaluate how close the predictions are to the true values in the validation set
###########################################################


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # sing R version 4.0.4`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation  <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#We divide into two sets: training and test sets
set.seed(1, sample.kind="Rounding")
test_index  <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
training_dataset <- edx[-test_index,]
temp <- edx[test_dataset,]

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

temp <- movielens[test_index,]

dim(training_dataset) #training set has 8,100,048 records


#We ensure that the userId and movieId are in both training and test sets
test_dataset <- temp %>%
  semi_join(training_dataset, by = "movieId") %>%
  semi_join(training_dataset, by = "userId") 

#We add the rows removed from or test dataset into training set
removed_rows <- anti_join(temp, test_dataset)
training_dataset <- rbind(training_dataset, removed_rows)
dim(test_dataset)#test set has 1,000,003 records
rm(test_index, temp, removed_rows)



##EXPLORATORY ANALYSIS##
head(edx) #Preview edx
anyNA(edx) #check missing values - results to FALSE

#We have 69,878 users and 10,677 movies in the edx set
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

#summary of edx
summarize(edx)

#view data types of edx
glimpse(edx, width = 50)

# We have 10 ratings from 0.5 to 5.0 with increments of 0.5
unique(edx$rating) #ratings used on edx

#Movie ratings
#Pulp Fiction (1994), Forrest Gump (1994) and Silence of the Lambs (1991) are the highest rated in that order
##plotting top 10 most popular movies
edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(10, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill = "brown", stat = "identity") +
  ggtitle("Top 10 most popular movies")+
  xlab("Count of ratings") +
  ylab(NULL) 

mean = mean(edx$rating)
mean #mean

# We visualize the training set rating distribution
edx %>% 
ggplot(aes(rating, y = ..prop..)) +
  geom_bar(fill = "brown") +
  ggtitle("Training set rating distribution")+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5))+ 
  labs(x = "Movie rating", y = "No. of ratings")

#We plot the no. of ratings by users
edx %>% 
  count(userId) %>% 
  ggplot(aes(n, fill = "brown")) + 
  geom_histogram( bins=30, color="black", show.legend = FALSE) +
  scale_x_log10() +
  labs(x = "Users", y = "Number of ratings")+
  ggtitle("Number of ratings by users")+
  theme(plot.title = element_text(hjust = 0.5))


#We plot mean rating by users
edx %>% group_by(userId) %>%
  summarise(mean_rating = sum(rating)/n()) %>%
  ggplot(aes(mean_rating, fill = "brown")) +
  geom_histogram( bins=30, color="black", show.legend = FALSE) +
  labs(x = "Average rating", y = "Number of users")+
  ggtitle("Average ratings by users")+
  theme(plot.title = element_text(hjust = 0.5))


#We have the release year from the  title into a separate column
edx <- edx %>% mutate(release_year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))

#Plot for average rating through the years
edx %>% group_by(release_year) %>%
  summarise(n =n(), avg = mean(rating)) %>%
  ggplot(aes(release_year, avg))+
  geom_point()+geom_hline(yintercept = mean, color = "red")+labs(x="Release year",y="Average rating")+theme(axis.text = element_text(size=12,face = "bold"))+
  ggtitle('Average rating through the years')+
  theme(plot.title = element_text(hjust = 0.5))

#count of ratings over years
edx %>% group_by(release_year) %>%
  summarise(ratings = n()) %>%
  ggplot(aes(release_year, ratings)) +
  geom_bar(stat = "identity", fill = "gray1", color = "white")+
  scale_y_continuous(labels = comma)+labs(x="Release year",y="no_rating")+
  theme(axis.text = element_text(size=12,face = "bold"))


#Top 3 movie genres which were highly reviewed
#Drama, Comedy, and Comedy|Romance
top_genres <- edx %>% group_by(genres) %>% 
  summarize(count = n()) %>% arrange(desc(count)) %>% head(3)
top_genres %>% knitr::kable()

#We summarize the popular genres by year
genres_by_year <- edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  select(movieId, release_year, genres) %>% 
  group_by(release_year, genres) %>% 
  summarise(count = n()) %>% arrange(desc(release_year)) 
genres_by_year

# Different periods show certain genres being more popular during those periods
# It is for this reason that we will not include genre into our prediction
ggplot(genres_by_year, aes(x = release_year, y = count)) + 
  geom_col(aes(fill = genres), position = 'dodge') + 
  ylab('No. of movies') + 
  xlab('Release year') +
  ggtitle('Popularity/year by genre')+
  theme(plot.title = element_text(hjust = 0.5))


###WE BEGIN THE PREDICTION APPROACH##

#We evaluate the accuracy using the RMSE
#RMSE measures the difference between predicted and observed values
#Our goal is to reduce the error below 0.8649

#We write a function to compute RMSE. 
rmse <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#First Model (naive model)
#We make a prediction using the mean of the movie ratings
#Also known as the naive model 
#It assumes the movies will have the same rating regardless of genre or user
#The mean obtained is 3.512413
mean_movie_rating <- mean(training_dataset$rating)
mean_movie_rating #mean rating

#We obtain our first RMSE = 1.060242
#The error is greater than 1 hence lacks accuracy. 
#Our Second Model will attempt to improve the error.
first_rmse <- rmse(test_dataset$rating, mean_movie_rating)

#Save RMSE result in a dataframe
results = data_frame(method = "Model 1: Using the mean (Naive model)", RMSE = first_rmse)
results %>% knitr::kable()

##Second Model (movie bias)
#We can confirm that some movies are more popular than others hence creating a bias
#We include bias_1 to represent the mean rating of the movies
movie_bias <- training_dataset %>%
  group_by(movieId) %>%
  summarize(bias_1 = mean(rating - mean_movie_rating))

#We plot movie bias
movie_bias %>% ggplot(aes(bias_1)) +
  geom_histogram(color = "black", fill = "brown", bins = 20) +
  xlab("Movie bias") +
  ylab("Count (n)")+
  ggtitle("Movie bias")

#Testing RMSE by adding the movie bias to our second model
#There is a slight improvement on our second model
predictions <- mean_movie_rating + test_dataset %>%
  left_join(movie_bias, by = "movieId") %>%
  pull(bias_1)
second_rmse <- rmse(predictions, test_dataset$rating)
results <- bind_rows(results,
                          data_frame(method="Model 2: Mean + movie bias",
                                     RMSE = second_rmse))
results %>% knitr::kable()

##Third Model (movie & user biases)
#Users can add bias by rating some movies very highly 
#Rating others very low or very high adds this bias to our model
#RMSE = 0.86397
#Having included movie and user biases the error is slightly lower
users_bias <- training_dataset %>%
  left_join(movie_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarize(bias_2 = mean(rating - mean_movie_rating - bias_1))

#We plot movie + User bias
users_bias %>% ggplot(aes(bias_2)) +
  geom_histogram(color = "black", fill = "brown", bins = 20) +
  xlab("User and movie bias") +
  ylab("Count (n)")+
  ggtitle("Movie + User bias")+
  theme(plot.title = element_text(hjust = 0.5))

predictions <- test_dataset %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(users_bias, by = "userId") %>%
  mutate(new_pred = mean_movie_rating + bias_1 + bias_2) %>%
  pull(new_pred)
  third_rmse <- RMSE(predictions, test_dataset$rating)
  results <- bind_rows(results,
                       data_frame(method="Model 3: Mean + movie + user bias",
                                  RMSE = third_rmse))
  results %>% knitr::kable()
  
  ##Fourth Model (regularized movie & user biases)
  #Some movies are rated by very few users, this can increase RMSE
  #Regularisation allows for reduced errors caused by movies 
  #with few ratings which can influence the prediction  and skew the error
  
  lambdas <- seq(from=0, to=10, by=0.25)
  rmses <- sapply(lambdas, function(x){
    
    #Adjust mean by movie effect and penalize low number on ratings
    movie_bias <- training_dataset %>%
      group_by(movieId) %>%
      summarize(movie_bias = sum(rating - mean_movie_rating)/(n()+x)) 
    
    #Adjust mean by user + movie effect and penalize low number of ratings
    user_bias <- training_dataset %>%
      left_join(movie_bias, by = "movieId") %>%
      group_by(userId) %>% 
      summarize(user_bias = sum(rating - movie_bias - mean_movie_rating)/(n()+x))
    
    #predict ratings in the training set to derive optimal penalty value 'lambda'
    predictions <- test_dataset %>%
      left_join(movie_bias, by = "movieId") %>%
      left_join(user_bias, by = "userId") %>%
      mutate(new_pred = mean_movie_rating + movie_bias + user_bias) %>%
      pull(new_pred)
    return(rmse(predictions, test_dataset$rating))
  })
  #We plot RMSE against lambdas to obtain optimal lambda
  qplot(lambdas, rmses, color = I("brown")) 

 #We apply lamda on Validation set
  lamd <- lambdas[which.min(rmses)]
  lamd #best lamda
  
  movie_bias <- edx %>% 
    group_by(movieId) %>%
    summarize(movie_bias = sum(rating - mean_movie_rating)/(n()+lamd))
  
  #regularize user bias
  user_bias <- edx %>% 
    left_join(movie_bias, by="movieId") %>%
    group_by(userId) %>%
    summarize(user_bias = sum(rating - movie_bias - mean_movie_rating)/(n()+lamd))
  
  predictions <- test_dataset %>% 
    left_join(movie_bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    mutate(new_pred = mean_movie_rating + movie_bias + user_bias) %>%
    pull(new_pred)
  fourth_rmse <- rmse(predictions, test_dataset$rating)
  results <- bind_rows(results,
                       data_frame(method="Model 4: Mean + movie + user bias + Regularisation",
                                  RMSE = fourth_rmse))
  results %>% knitr::kable()
  
  
  
  ##CONCLUSIONS## 
  #We employed a 4-step approach to come up with a movie recommendation model. 
  #The final model (fourth model) is the preferred recommendation model since its RMSE is below the target of 0.8649.

  