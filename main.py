import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from flask import Flask, render_template

app = Flask(__name__)

# Ensure the static directory exists
static_dir = os.path.join(os.getcwd(), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Load the dataset
nutri = pd.read_csv('nutrition_elderly.csv')

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

# Save the first plot as an image file in the static directory
plot1 = os.path.join('static', 'nutrition_plot1.png')
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

# Save the second plot as an image file in the static directory
plot2 = os.path.join('static', 'nutrition_plot2.png')
plt.savefig(plot2)
plt.close()

# Create the third scatter plot between situation and coffee
plt.scatter(nutri['situation'], nutri['coffee'], edgecolor='black')
plt.title('Situation vs Coffee')
plt.xlabel('Situation')
plt.ylabel('Coffee Consumption')

# Save the third plot as an image file in the static directory
plot3 = os.path.join('static', 'nutrition_plot3.png')
plt.savefig(plot3)
plt.close()

# Create the fourth scatter plot between age and height
plt.scatter(nutri['age'], nutri['height'], edgecolor='black', alpha=0.7)
plt.title('Age vs Height')
plt.xlabel('Age')
plt.ylabel('Height')

# Save the fourth plot as an image file in the static directory
plot4 = os.path.join('static', 'nutrition_plot4.png')
plt.savefig(plot4)
plt.close()

# Create the fifth boxplot of Age
plt.boxplot(nutri['age'], widths=width, vert=False)
plt.title('Boxplot Age')
plt.xlabel('Age')

# Save the fifth plot as an image in the static directory
plot5 = os.path.join('static', 'nutrition_plot5.png')
plt.savefig(plot5)
plt.close()

# Create the sixth boxplot of height
plt.boxplot(nutri['height'], widths=width, vert=False)
plt.title('Boxplot Height')
plt.xlabel('Height')

# Save the sixth plot as an image in the static directory
plot6 = os.path.join('static', 'nutrition_plot6.png')
plt.savefig(plot6)
plt.close()

# Create the seventh Histogram of Age
weights = np.ones_like(nutri.age) / nutri.age.count()
plt.hist(nutri.age, bins = 9, weights = weights, facecolor = 'cyan', edgecolor = 'black', linewidth = 1)
plt.xlabel('Age')
plt.ylabel('Proportion of Total')

# Save the seventh Histogram of Age
plot7 = os.path.join('static', 'nutrition_plot7.png')
plt.savefig(plot7)
plt.close()

# Create the eigth Empirical distribution
x = np.sort(nutri.age)
y = np.linspace(0,1,len(nutri.age))
plt.xlabel('age')
plt.ylabel('Fn(x)')
plt.step(x,y)
plt.xlim(x.min(), x.max())

#Save the eight Emperical distribution
plot8 = os.path.join('static', 'nutrition_plot8.png')
plt.savefig(plot8)
plt.close()

#Create the first Bivariate Plot(Qualitative Data)
sns.countplot(x='situation',hue='gender', data=nutri,hue_order=['Male', 'Female'],palette=['SkyBlue','Pink'], saturation=1, edgecolor='black')
plt.legend(loc='upper center')
plt.xlabel('')
plt.ylabel('Counts')

#Save the first Bivariate Plot 
plot9 = os.path.join('static', 'nutrition_plot9.png')
plt.savefig(plot9)
plt.close()

#Create the second Bivariate Data (Quantitative Data)
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

#Save the second Bivarite Data
plot10 = os.path.join('static','nutrition_plot10.png')
plt.savefig(plot10)
plt.close()

#Create the the plot for One Qualitative and One Quantitative Variable
males = nutri[nutri.gender =='Male']
females = nutri[nutri.gender =='Female']
plt.boxplot([males.weight, females.weight], notch=True, widths=(0.5,0.5))
plt.xlabel('gender')
plt.ylabel('weight')
plt.xticks([1,2],['Male','Female'])

#Save the plot for One Qualitative and One Quantitative
plot11 = os.path.join('static','nutrition_plot11.png')
plt.savefig(plot11)
plt.close()

@app.route('/')
def index():
    # Convert the DataFrame to an HTML table
    nutri_html = nutri.to_html(classes='table table-striped', index=False)
    # Render the HTML template and pass the plot filenames and the HTML table
    return render_template('index.html', plot1_url=plot1, plot2_url=plot2, plot3_url=plot3, plot4_url=plot4, plot5_url=plot5, plot6_url=plot6, plot7_url=plot7, plot8_url=plot8,plot9_url=plot9, plot10_url=plot10,plot11_url=plot11, table=nutri_html)

if __name__ == '__main__':
    app.run(debug=True)
 