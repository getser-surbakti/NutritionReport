import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from flask import Flask, render_template, send_from_directory
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the writable directory for saving plots
writable_dir = "/tmp"

# Load the dataset
try:
    nutri = pd.read_csv('nutrition_elderly.csv')
    logging.debug("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("The dataset file was not found.")
    nutri = pd.DataFrame()

# Check if the dataframe is loaded correctly
if not nutri.empty:
    # Removing Column
    nutri = nutri.drop('fish', axis=1)

    # Replace and convert columns
    DICT = {1: 'Male', 2: 'Female'}
    nutri['gender'] = nutri['gender'].replace(DICT).astype('category')
    nutri['height'] = nutri['height'].astype(float)
    DICT1 = {1: 'Single', 2: 'Living with spouse', 3: 'Living with family', 4: 'Living with someone else'}
    nutri['situation'] = nutri['situation'].replace(DICT1).astype('category')
    nutri['age'] = nutri['age'].astype(float)

    # Calculate the counts for each situation
    situation_counts = nutri['situation'].value_counts()

    # Define the positions and width for the bars
    x = range(len(situation_counts))
    width = 0.35

    # Create the first bar plot
    plt.bar(x, situation_counts, width, edgecolor='black')
    plt.xticks(x, situation_counts.index)
    plt.title('Situation Counts')
    plt.xlabel('Situation')
    plt.ylabel('Counts')

    # Save the first plot as an image file in the writable directory
    plot1 = os.path.join(writable_dir, 'nutrition_plot1.png')
    plt.savefig(plot1)
    plt.close()

    # Create the second plot (e.g., gender distribution)
    gender_counts = nutri['gender'].value_counts()
    x2 = range(len(gender_counts))
    plt.bar(x2, gender_counts, width, edgecolor='black', color='orange')
    plt.xticks(x2, gender_counts.index)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Counts')

    # Save the second plot as an image file in the writable directory
    plot2 = os.path.join(writable_dir, 'nutrition_plot2.png')
    plt.savefig(plot2)
    plt.close()

    # Create the third scatter plot between situation and coffee
    plt.scatter(nutri['situation'], nutri['coffee'], edgecolor='black')
    plt.title('Situation vs Coffee')
    plt.xlabel('Situation')
    plt.ylabel('Coffee Consumption')

    # Save the third plot as an image file in the writable directory
    plot3 = os.path.join(writable_dir, 'nutrition_plot3.png')
    plt.savefig(plot3)
    plt.close()

    # Create the fourth scatter plot between age and height
    plt.scatter(nutri['age'], nutri['height'], edgecolor='black', alpha=0.7)
    plt.title('Age vs Height')
    plt.xlabel('Age')
    plt.ylabel('Height')

    # Save the fourth plot as an image file in the writable directory
    plot4 = os.path.join(writable_dir, 'nutrition_plot4.png')
    plt.savefig(plot4)
    plt.close()

    # Create the fifth boxplot of Age
    plt.boxplot(nutri['age'], widths=width, vert=False)
    plt.title('Boxplot Age')
    plt.xlabel('Age')

    # Save the fifth plot as an image in the writable directory
    plot5 = os.path.join(writable_dir, 'nutrition_plot5.png')
    plt.savefig(plot5)
    plt.close()

    # Create the sixth boxplot of height
    plt.boxplot(nutri['height'], widths=width, vert=False)
    plt.title('Boxplot Height')
    plt.xlabel('Height')

    # Save the sixth plot as an image in the writable directory
    plot6 = os.path.join(writable_dir, 'nutrition_plot6.png')
    plt.savefig(plot6)
    plt.close()

    # Create the seventh Histogram of Age
    weights = np.ones_like(nutri.age) / nutri.age.count()
    plt.hist(nutri.age, bins=9, weights=weights, facecolor='cyan', edgecolor='black', linewidth=1)
    plt.xlabel('Age')
    plt.ylabel('Proportion of Total')

    # Save the seventh Histogram of Age
    plot7 = os.path.join(writable_dir, 'nutrition_plot7.png')
    plt.savefig(plot7)
    plt.close()

    # Create the eighth Empirical distribution
    x = np.sort(nutri.age)
    y = np.linspace(0, 1, len(nutri.age))
    plt.xlabel('Age')
    plt.ylabel('Fn(x)')
    plt.step(x, y)
    plt.xlim(x.min(), x.max())

    # Save the eighth Empirical distribution
    plot8 = os.path.join(writable_dir, 'nutrition_plot8.png')
    plt.savefig(plot8)
    plt.close()

    # Create the first Bivariate Plot (Qualitative Data)
    sns.countplot(x='situation', hue='gender', data=nutri, hue_order=['Male', 'Female'], palette=['SkyBlue', 'Pink'], saturation=1, edgecolor='black')
    plt.legend(loc='upper center')
    plt.xlabel('')
    plt.ylabel('Counts')

    # Save the first Bivariate Plot 
    plot9 = os.path.join(writable_dir, 'nutrition_plot9.png')
    plt.savefig(plot9)
    plt.close()

    # Create the second Bivariate Data (Quantitative Data)
    plt.figure(figsize=(8, 6))
    plt.scatter(nutri.height, nutri.weight, s=50, c='blue', marker='o', edgecolor='black', alpha=0.7)
    plt.xlabel('Height (cm)', fontsize=14)
    plt.ylabel('Weight (kg)', fontsize=14)
    plt.title('Height vs. Weight Scatter Plot with Line of Best Fit', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a line of best fit
    m, b = np.polyfit(nutri.height, nutri.weight, 1)  # Linear fit
    plt.plot(nutri.height, m*nutri.height + b, color='red', linewidth=2)

    # Save the second Bivariate Data
    plot10 = os.path.join(writable_dir, 'nutrition_plot10.png')
    plt.savefig(plot10)
    plt.close()

    # Create the plot for One Qualitative and One Quantitative Variable
    males = nutri[nutri.gender == 'Male']
    females = nutri[nutri.gender == 'Female']
    plt.boxplot([males.weight, females.weight], notch=True, widths=(0.5, 0.5))
    plt.xlabel('Gender')
    plt.ylabel('Weight')
    plt.xticks([1, 2], ['Male', 'Female'])

    # Save the plot for One Qualitative and One Quantitative
    plot11 = os.path.join(writable_dir, 'nutrition_plot11.png')
    plt.savefig(plot11)
    plt.close()

# Serve the images from the writable directory
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(writable_dir, filename)

@app.route('/')
def index():
    if not nutri.empty:
        # Convert the DataFrame to an HTML table
        nutri_html = nutri.to_html(classes='table table-striped', index=False)
        # Render the HTML template and pass the plot filenames and the HTML table
        return render_template('index.html', plot1_url='/images/nutrition_plot1.png', plot2_url='/images/nutrition_plot2.png', plot3_url='/images/nutrition_plot3.png', plot4_url='/images/nutrition_plot4.png', plot5_url='/images/nutrition_plot5.png', plot6_url='/images/nutrition_plot6.png', plot7_url='/images/nutrition_plot7.png', plot8_url='/images/nutrition_plot8.png', plot9_url='/images/nutrition_plot9.png', plot10_url='/images/nutrition_plot10.png', plot11_url='/images/nutrition_plot11.png', table=nutri_html)
    else:
        return "Error: Dataset could not be loaded."

if __name__ == '__main__':
    app.run(debug=True)
