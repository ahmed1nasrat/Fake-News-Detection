
library(e1071)
library(tm)
library(textTinyR)
library(caret)
library(ggplot2)  
library(dplyr)
library(tidyr)  
library(caTools)
library(pROC)
library(class)
  # loading data
    setwd("D:/project stat")
    df <- read.csv("news.csv") 
    # giving a look on the data
    print(head(df,n=1))
    print("coulmns names: ")
    print(colnames(df))
    
    # Preprocessing 
  
    #Checking duplicates
    if(any(duplicated(df))){
      print("there is duplicates")
    }
    
    #Removing duplicates 
    print("number of rows before removing duplicates")
    print (nrow(df))
    df<-df[!duplicated(df),]
    print("number of rows after removing duplicates")
    print (nrow(df))
  
    #Relabeling data
    df$label[df$label == "F"]<-"FAKE"
    df$label[df$label == "R"]<-"REAL"
  
    df$label=as.factor(df$label)
    print(unique(df$label))
    
    
    #Checking null values
    print("number of rows with null values")
    print(sum(!complete.cases(df)))
  
    print(summary(df))
    #Plot labels
   plot(df$label,xlab="Labels",col="blue",ylab="Count")
    
    #Creating corpus 
    corpus_title <- Corpus(VectorSource(df$title))
    corpus_text <- Corpus(VectorSource(df$text))
    
    # Preprocess the text data for 'title'
    corpus_title <- tm_map(corpus_title, content_transformer(tolower))
    corpus_title <- tm_map(corpus_title, removePunctuation)
    corpus_title <- tm_map(corpus_title, removeNumbers)
    corpus_title <- tm_map(corpus_title, removeWords, stopwords("en"))
    corpus_title <- tm_map(corpus_title, stripWhitespace)
    #corpus_title <- tm_map(corpus_title, stemDocument)
    # Preprocess the text data for 'text'
    corpus_text <- tm_map(corpus_text, content_transformer(tolower))
    corpus_text <- tm_map(corpus_text, removePunctuation)
    corpus_text <- tm_map(corpus_text, removeNumbers)
    corpus_text <- tm_map(corpus_text, removeWords, stopwords("en"))
    corpus_text <- tm_map(corpus_text, stripWhitespace)
    corpus_text <- tm_map(corpus_text, stemDocument)
    # Create a document-term matrix "DTM" for 'title'
    dtm_title <- DocumentTermMatrix(corpus_title)
    dtm_text <- DocumentTermMatrix(corpus_text)
    # Find frequents words for 'title'
    freq_words_title <- findFreqTerms(dtm_title, 65)
    freq_words_text <- findFreqTerms(dtm_text,1700)
    freq_words_text=setdiff(freq_words_text,intersect(freq_words_text,freq_words_title))
    dtm_title <- dtm_title[, freq_words_title]
    dtm_text <- dtm_text[, freq_words_text]

  
    # Convert data into 0 ,1
    convert_counts <- function(x) {
      x <- ifelse(x > 0, 1,0 )
    }
     # merge text & title
    freq_train<- cbind(dtm_text,dtm_title)
 
    
    #convert to 0 & 1
    
    freq_train <- apply(freq_train, MARGIN = 2, FUN = convert_counts)
      
    
    #convert to data frame
    
    data=as.data.frame(freq_train)
    set.seed(23)
    train_indices <- sample.split(data, SplitRatio = 0.8)
    train_data <- data[train_indices, ]
    test_data <- data[!train_indices, ]
    # Labels
    train_labels <- df[train_indices,]$label
    test_labels <- df[!train_indices,]$label
    
    train_data$labels=train_labels
  
    test_data$labels=test_labels
    
    #get top 10 frequent words
    
    freq=colSums(subset(train_data,select = -labels))
    freq=sort(freq,decreasing = TRUE)
    print(names(head(freq,10)))
    plot_data = train_data[,names(head(freq,10))]
    plot_data$label=train_data$labels
    #Converting data to vertical
    data_long <- pivot_longer(data = plot_data,cols=-label, names_to = "column", values_to = "value") 
    data_long<-data_long[data_long$value==1,]
    #Ploting
    ggplot(data_long, aes(x = column, fill = label)) +  
      geom_bar(position = "dodge") +  
      labs(x = "Column", y = "Count", fill = "Label") 
    
 
    # Naive Bayes
    np <- naiveBayes(labels~.,data = train_data)
    pred_train <- predict(np,train_data)
    confusion_matrix_train <- table(pred_train, train_data$labels)
    print(confusion_matrix_train)
  
    accuracy_combined <- sum(diag(confusion_matrix_train)) / sum(confusion_matrix_train)
    print(paste("Naive Bayes train  Accuracy:", accuracy_combined))
    
    pred_test <- predict(np, test_data)
    confusion_matrix_test <- table(pred_test, test_data$ labels)
    print(confusion_matrix_test)
  
    accuracy_combined <- sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test)
    print(paste("Naive Bayes test  Accuracy:", accuracy_combined))
  
    #Roc curve
    
    roc_curve <- roc(test_labels,as.numeric( pred_test))
    plot(roc_curve, print.auc = TRUE,main="naive bayes roc curve")
    
    
    
    # Logistic Regression Model
    
    
    
    lg <- glm( labels ~ .,data = train_data, family = binomial())
    predictions <- predict(lg, newdata = train_data, type = "response")
    predictions <- ifelse(predictions > 0.5, 1, 0)
    
     #Confusion Matrix
    conf_matrix <- table(predictions, train_data$ labels)
    print(conf_matrix)
    
    # Calculate Accuracy
    accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
    print(paste("logistic train Accuracy:", accuracy))
    
    
    
    # Predict using the correct model and processed test data
    
    predictions <- predict(lg, newdata = test_data, type = "response")
    
    # Convert predicted probabilities to binary predictions
    predictions <- ifelse(predictions > 0.5, 1, 0)
    
    # Create confusion matrix
    conf_matrix <- table(predictions, test_data$ labels)
    print(conf_matrix)
    
    acc <- sum(diag(conf_matrix)) / sum(conf_matrix)
    print(paste("logistic test Accuracy:", acc))
    #roc_curve
    roc_curve <- roc(test_labels, predictions)
    plot(roc_curve, print.auc = TRUE,main="logistic regretion roc curve")
    
    # SVM Model
    svm_model <- svm(  labels~.,train_data)
    svm_pred <- predict(svm_model, train_data,)
    #svm_pred <- ifelse(svm_pred > 0.5, 1, 0)
    # Confusion Matrix
    conf_matrix_svm <- table(svm_pred, train_data$ labels)
    
    print(conf_matrix_svm)
    
    # Calculate Accuracy
    accuracy_svm <- sum(diag(conf_matrix_svm)) / sum(conf_matrix_svm)
    print(paste("SVM train Accuracy:", accuracy_svm))
    
    # Predictions
    svm_pred <- predict(svm_model, test_data)
    #svm_pred<- ifelse(svm_pred > 0.5, 1, 0)
    # Confusion Matrix
    conf_svm <- table(svm_pred, test_data$ labels)
    
    print(conf_svm)
    
    # Calculate Accuracy
    accuracy <- sum(diag(conf_svm)) / sum(conf_svm)
    print(paste("SVM test Accuracy:", accuracy))
    roc_curve <- roc(test_labels,as.numeric( svm_pred))
    plot(roc_curve, print.auc = TRUE,main="SVM roc curve")
    # Load kNN library
    

    
    knn_train=subset(train_data, select = -labels)
    knn_test=subset(test_data, select = -labels)
    knn_model <- knn(knn_train, knn_test, train_data$labels, k = 3)
    conf_matrix_train <- table(knn_model, test_data$labels)
    print(conf_matrix_train)
    accuracy_train <- sum(diag(conf_matrix_train)) / sum(conf_matrix_train)
    print(paste("KNN  Accuracy:", accuracy_train))
    roc_curve <- roc(test_labels,as.numeric( knn_model))
    plot(roc_curve, print.auc = TRUE,main="knn roc curve")
