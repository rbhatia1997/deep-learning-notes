# Introduction
This will contain concepts and coding tools used for the fourth week of the tutorial. 

## Implementing the YOLO Algorithm 

In this assignment, we had to build a car detection system. We took all the images in a folder and drew bounding boxes. We represent the classes with some integer or vector. We are using YOLO to do this. Input is a batch of images with a shape m, 608, 608, 3 and the output is a list of bounding boxes with recognized classes. Anchor boxes are chosen by exploring the training data and chosing reasonable ratios. This data is given to us and the dimensions are m, nh, nw, anchors, classes. The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85) in this case. 

We have five anchor boxes so each of the 19x19 cells gives info about five boxes and if we flatten the last two elements, the output of the deep CNN is therefore 19, 19, and 425 (which is basically 5 boxes x 85 classes). The class score (for each box of each cell) is found by element wise product and then we get a probability from that. Score is PC x CI (probability there's PC by probability it's from a certain class). You assign this class score (the maximum one) to the box. You take a max across the 80 classes, one max for each five anchors. 

In the image above, we just plot boxes for high probability but there's still too many boxes; we want to reduce the output to a much smaller number of detected objects. Boom let's utilize non-max suppression. You get rid of low-score boxes (not sure about detecting a class or low probability) and select only one box when several boxes have overlap. 

The first step is filter by thresholding, you get rid of any box where the class score is less than a given threshold. The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It is convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:box_confidence: 

* tensor of shape  (19×19,5,1)(19×19,5,1)  containing  pcpc  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
* boxes: tensor of shape  (19×19,5,4)(19×19,5,4)  containing the midpoint and dimensions  (bx,by,bh,bw)(bx,by,bh,bw)  for each of the 5 boxes in each cell.
* box_class_probs: tensor of shape  (19×19,5,80)(19×19,5,80)  containing the "class probabilities"  (c1,c2,...c80)(c1,c2,...c80)  for each of the 80 classes for each of the 5 boxes per cell.

So the first thing was yolo_filter_boxes that computed box scores via element wise product (literally multiply) because of broadcasting. For each box, then, we found the index and corresponding box score for the max (used argsMax and max) and then applied the mask. This looks like the following:

'''python
box_scores = box_confidence * box_class_probs # element wise product 

box_classes = K.argmax(box_scores, axis=-1) # selecting last element. argmax. 
box_class_scores = K.max(box_scores, axis=-1)

filtering_mask = ((box_class_scores) >= threshold)

scores = tf.boolean_mask(box_class_scores, filtering_mask, name='boolean_mask')
boxes = tf.boolean_mask(boxes, filtering_mask, name='boolean_mask')
classes = tf.boolean_mask(box_classes, filtering_mask, name='boolean_mask')
'''

Now, even after filtering by thresholding over class scores, we have too many boxes. Time for non-max suppression. Non max suppression uses intersection over union to select most accurate bounding box. Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B). 

Now from here, we can implement non-max suppression by selecting the highest score, computing overlap with other boxes and removing (iou > threshold) and then iterating. Can do this via ```nms_indices = tf.image.non_max_suppression(boxes = boxes, scores = scores, max_output_size = max_boxes, iou_threshold = iou_threshold)```

Finally, you impelement something that takes the output of the deep CNN and filtering through the boxes using the functions implemented. Basically it retrieves YOLO output, converts boxes to be ready for filtering functions, uses yolo_filter_boxes to do score filtering, then scales boxes to original image shape, and uses the non-max suppression function to show bounding boxes. 

You define class names, anchors, image shape, and session. We take a pre-trained Keras YOLO model and load it. We then convert the output of the model to usable bounding box tensors via yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names)). Then we filter and select best boxes via yolo_eval.  

* yolo_model.input is given to yolo_model. The model is used to compute the output yolo_model.output
* yolo_model.output is processed by yolo_head. It gives you yolo_outputs
* yolo_outputs goes through a filtering function, yolo_eval. It outputs your predictions: scores, boxes, classes. 
