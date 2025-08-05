#!/bin/bash
# Script to test all example API calls from prompt.txt using curl

API_URL="http://localhost:8000/api/"

# Example 2: IMDb Ratings for Top Movies
echo "Testing IMDb Ratings for Top Movies..."
cat > imdb_questions.txt <<EOF
Scrape IMDb Ratings for Top Movies
Scrape the top 50 movies from:
https://www.imdb.com/chart/top
Then:
Extract movie name, year, rating.
Create a histogram of IMDb ratings.
Answer:
What is the average rating?
Which decade has the most top-rated movies?
EOF
echo "Testing IMDb Ratings for Top Movies..." >> test_examples_output.txt
curl -s -X POST "$API_URL" -F "questions_txt=@imdb_questions.txt" >> test_examples_output.txt
echo -e "\n" >> test_examples_output.txt

# Example 3: Inflation Data from Trading Economics
echo "Testing Inflation Data from Trading Economics..."
cat > inflation_questions.txt <<EOF
Scrape Inflation Data from Trading Economics
Instruction to Agent:
Scrape inflation rate data for India from:
https://tradingeconomics.com/india/inflation-cpi
Then:
Plot a time series of inflation over the last 12 months.
Answer:
What is the current inflation rate?
What was the highest rate in the last year?
EOF
echo "Testing Inflation Data from Trading Economics..." >> test_examples_output.txt
curl -s -X POST "$API_URL" -F "questions_txt=@inflation_questions.txt" >> test_examples_output.txt
echo -e "\n" >> test_examples_output.txt

# Example 4: COVID-19 Cases Data
echo "Testing COVID-19 Cases Data..."
cat > covid_questions.txt <<EOF
Scrape COVID-19 Cases Data
Instruction to Agent:
Scrape the table from:
https://www.worldometers.info/coronavirus/
Then:
Extract top 20 countries by total cases.
Plot total cases vs. deaths (scatter plot).
Answer:
Which country has the highest death-to-case ratio?
What's the global average recovery rate?
EOF
echo "Testing COVID-19 Cases Data..." >> test_examples_output.txt
curl -s -X POST "$API_URL" -F "questions_txt=@covid_questions.txt" >> test_examples_output.txt
echo -e "\n" >> test_examples_output.txt


# Cleanup temp files
rm -f imdb_questions.txt inflation_questions.txt covid_questions.txt
echo "All example tests completed."
# Cleanup only generated _questions.txt files
rm -f imdb_questions.txt inflation_questions.txt covid_questions.txt cricket_questions.txt

echo "All example tests completed. Output logged to test_examples_output.txt."
