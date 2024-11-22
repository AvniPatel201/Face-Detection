To do: 


  
3. Performance Evaluation
   You need to compare and evaluate the performance of the implemented face detection and emotion recognition methods.

    Tasks:

      Accuracy Comparison:
         Compare detection accuracy across methods under normal and challenging conditions (e.g., varying lighting, shadows).
         Measure false positives and false negatives.
      Runtime Analysis:
         Record and compare the runtime of each method for face detection and emotion recognition.
         Use Python’s time library to measure execution time.
      Vary Feature Points:
         For SIFT/ORB, experiment with varying numbers of key points to observe their impact on detection accuracy and speed.
   
4. Experiment with Different Lighting Conditions
   Test your implementations with images that simulate:
        Strong shadows
        Low light
        Bright light or overexposure
        Occluded faces (e.g., wearing glasses, hats, or masks)
   
4. Visualize Results
   Generate visual comparisons and performance summaries:

   Runtime Graphs:
       Use libraries like Matplotlib or Seaborn to plot runtime differences across methods.
       Show trends for detection speed versus feature points.
   Accuracy Charts:
       Visualize detection accuracy under different lighting or occlusion scenarios.
       Create bar or line graphs to compare the performance of the methods.
   
5. Document Results
  Create a summary of your findings:

    Discuss the strengths and weaknesses of each detection and emotion recognition method.
    Highlight which methods are better suited for specific challenging conditions.
   
6. Test Robustness of Emotion Recognition
    Your emotion recognition code works, but you should:

    Evaluate the accuracy of the detected emotions under challenging conditions.
    Compare the emotion recognition results for images with multiple people or occlusions.
