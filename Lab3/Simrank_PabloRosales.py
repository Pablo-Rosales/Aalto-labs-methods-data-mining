import pandas as pd
import numpy as np
import math

# EXERCISE 4
# The initial part of the code computes the SimRank for the user chosen down
# below. Modify the user ID in the variable user (down below, line 99)

# Function for computing the simrank between jokes
def compute_simrank_jokes(simrank_users,rows,columns,c2,X):
    simrank_jokes=np.zeros((columns-2,columns-2))
    for j_row in range(columns-2):
        for j_column in range(columns-2):
            if j_row==j_column:
                simrank_jokes[j_row,j_column]=1
            else:
                sum_users=0
                in_neighbors_1=sum(X[:,j_row+1])
                in_neighbors_2=sum(X[:,j_column+1])
                for i in range(rows):
                    if X[i,j_row]>0:
                        for j in range(rows):
                            if X[j,j_column]>0:
                                sum_users=sum_users+simrank_users[i,j]
                                
                simrank_jokes[j_row,j_column]=c2/(in_neighbors_1*in_neighbors_2)*sum_users

    return simrank_jokes


# The data is loaded from the csv file
data = pd.read_csv("jester-800-10.csv")
data = data.drop(columns = ['user_id'])
# The data is transformed into an array
X = np.array(data)

# Read the test data from csv file
test = pd.read_csv("test-800-10.csv")
test = test.drop(columns = ['user_id'])
A = np.array(test)

X = np.vstack((A, X))




# The rows represent the in-neighbors and out-neighbors of the users X[1,:]
# The columns represent the in-neighbors and out-neighbors of the jokes X[:,2]

rows, columns = X.shape

# Iterations of SimRank algorithm
k = 5

# Simrank users is a square matrix that stores the simrank between users
simrank_users = np.zeros((rows,rows))
#print(simrank_users)
c1 = 0.8
c2 = 0.8

for iteration in range(k):
    print("\nIteration: ", iteration)
    if iteration==0:
        for u_row in range(rows):
            for u_column in range(rows):
                if u_row==u_column:
                    simrank_users[u_row,u_column]=1
                else:
                    simrank_users[u_row,u_column]=0       
        
    else:
        simrank_jokes = compute_simrank_jokes(simrank_users,rows,columns,c2,X)
        for u_row in range(rows):
            for u_column in range(rows):
                if u_row != u_column:
                    sum_jokes=0
                    for i in range(columns-1):
                        if i>0 and X[u_row,i]>0:
                            for j in range(columns-1):
                                if j>0 and X[u_column,j]>0:
                                    sum_jokes=sum_jokes+simrank_jokes[i-1,j-1]
                                    
                            
                
                    simrank_users[u_row,u_column]=c1/(X[u_row,columns-1]*X[u_column,columns-1])*sum_jokes
                
        
                    
np.savetxt("simrank_users.csv", simrank_users, delimiter=" ")            
               
print("\nEND")   

#########################################################################################################
# RECOMMENDATION ALGORITHM FOR THE CHOSEN USER
# Initially the most similar users are identified

# The variable number of users represents the number of similar users that are
# considered for performing the recommendation

# The user id can be chosen modifying the variable user
number_users = 50
user = 4519

if user in X[:,0]:
    list_X = X[:,0].tolist()
    row_X = list_X.index(user)
    row = simrank_users[row_X,:]
    number_recommendations = (columns-2)-X[row_X,columns-1]
    indices = (-row).argsort()[:number_users]
    recommendation=np.full((number_recommendations),100)
    rec = 0

    for j in range(number_users):
        if j>0 and rec<number_recommendations-1:
            for i in range(columns-2):
                if i>0 and rec<number_recommendations-1:
                    if X[indices[j],i]==1 and X[row_X,i]==0 and ((i-1) in recommendation)==False:
                        recommendation[rec]=i-1
                        rec = rec + 1
else:
    print("\nThe user does not exist")
                    
print(recommendation)                   
                
        
