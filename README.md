# Business Problem

Baseball with shared salary information and career statistics for 1986
A machine learning project was carried out for the salary estimates of the players

# Data Set Story
This dataset was originally taken from the StatLib library at Carnegie Mellon University.
The dataset is part of the data used in the 1988 ASA Graphics Section Poster Session.
Salary data is originally from Sports Illustrated, April 20, 1987. 1986 and career statistics are published by Collier Books, Macmillan Publishing Company, New York
Obtained from the 1987 Baseball Encyclopedia Update.

 AtBat: Number of hits with a baseball bat during the 1986-1987 season
 
 Hits: the number of hits in the 1986-1987 season
 
 HmRun: Most valuable hits in the 1986-1987 season
 
 Runs: The points he earned for his team in the 1986-1987 season
 
 RBI: The number of players a batter had jogged when he hit
 
 Walks: Number of mistakes made by the opposing player
 
 Years: Player's playing time in major league (years)
 
 CAtBat: Number of times the player hits the ball during his career
 
 CHits: The number of hits the player has made throughout his career
 
 CHmRun: The player's most valuable number during his career
 
 CRuns: The number of points the player has earned for his team during his career
 
 CRBI: The number of players the player has made during his career
 
 CWalks: The number of mistakes the player has made to the opposing player during his career
 
 League: A factor with levels A and N, showing the league in which the player played until the end of the season
 
 Division: a factor with levels E and W, showing the position played by the player at the end of 1986
 
 PutOuts: Helping your teammate in-game
 
 Assits: Number of assists by the player in the 1986-1987 season
 
 Errors: the number of errors of the player in the 1986-1987 season
 
 Salary: The salary of the player in the 1986-1987 season (over thousand)
 
 NewLeague: a factor with levels A and N indicating the league of the player at the beginning of the 1987 season
 
 # Conclusion
 
 Using the Linear Regression model, the cross validation score 175 was obtained. There was compelling loss in the data set with the deletion of outliers and Null 
 dependent variables. Outlier limits are lowered and the total number of data in the dataset is increased. However, since linear regression is highly affected by   
 outliers, the model has a higher score.
