mv regression_final_results_models.zip ./regression_final_results
cd regression_final_results
unzip regression_final_results_models.zip
mv regression_final_results_models/* .
rm -rf regression_final_results_models
cd ..

mv binary_final_results_models.zip ./binary_final_results
cd binary_final_results
unzip binary_final_results_models.zip
mv binary_final_results_models/* .
rm -rf binary_final_results_models

