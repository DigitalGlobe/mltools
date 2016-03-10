def main(job_file):
    """Runs the simple_lulc workflow.

       Args:
           job_file (str): Job filename (.json, see README of this repo) 
    """    
   
    # get job parameters
    job = json.load(open(job_file, 'r'))
    image_file = job["image_file"]
    train_file = job["train_file"]
    target_file = job["target_file"]
    output_file = job["output_file"]
    algo_params = job["params"]       # these are parameters pertinent to the 
                                      # algorithm
    
    # Using random forest classifier
    n_estimators = algo_params["n_estimators"]
    oob_score = algo_params["oob_score"]
    class_weight = algo_params["class_weight"]
    classifier = RandomForestClassifier(n_estimators = n_estimators, 
                                        oob_score = oob_score, 
                                        class_weight = class_weight)
        
    print "Train model"
    trained_classifier = train_model(train_file, image_file, classifier)
    
    print "Classify"
    labels, scores, priorities = classify_w_scores(target_file, image_file, 
                                                   trained_classifier)
    
    print "Write results"    
    values = zip(labels, scores, priorities)
    jt.write_values_to_geojson(values, 
                               ['class_name', 'score', 'tomnod_priority'], 
                               target_file, 
                               output_file)

    # Compute confusion matrix; this makes sense only if the target file
    # contains known labels
    print "Confusion matrix"
    C = jt.confusion_matrix_two_geojsons(target_file, output_file)
    print C

    print "Normalized confusion matrix"
    print C.astype(float)/C.sum(1)[:, None]

    print "Done!"
    

