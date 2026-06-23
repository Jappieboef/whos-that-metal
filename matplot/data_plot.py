import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Loads the CSV file containing the NWChem results into a DataFrame and prints it to the console to view.
df=pd.read_csv('data/nwchem_results.csv')
#print(df)

#this only grabs the columns that start with 'spec_' and stores them in a list called spec_column. It then sorts the list based on the integer value that comes after 'spec_' in the column name.
spec_column = []
for c in df.columns:
    if c.startswith('spec_'):
        spec_column.append(c)

# sorts them numerically based on the number that comes after 'spec_' in the column name. For example, 'spec_50' would come before 'spec_60' because 1 is less than 10.
spec_column.sort(key=lambda c: int(c.split('_')[1]))

#Builds the x-axis value of the plot by extracting the interger from the spec_
x=[]
for c  in spec_column:
    x.append(int(c.split('_')[1]))


#output directory for the plots png files that we are making.
output_dir="NWChem_Plots"

os.makedirs(output_dir, exist_ok=True)


for index, row in df.iterrows():
    #This is the y-axis and it grabs the intesity values for that acroos all the spec_columns
    y= row[spec_column].values
    
    #plots the wavenumber(x) againts the intensities(y) 
    plt.plot(x,y)

    #labels and titles for the plots
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensities')

    #This uses name for the plot title.
    plot_title=(row['filename'])
    plt.title(plot_title)

    #This removes the .csv from the file name so it can be saves as a png
    safefilename=(str(plot_title).replace(".csv",""))

    #This is the filepath for the plot to be saved in the output directory with the name of the plot title.
    filepath=os.path.join(output_dir,f"{safefilename}.png")

    #check to make sure png file does not already exist in the output directory. If it does, it will skip that plot and move on to the next one.
    #!!! Can be commented out if you want to overwrite the plots each time the script is run.
    if os.path.exists(filepath):
        print(f"Plot {safefilename}.png already exists in the {output_dir} folder. Skipping this plot.")
        continue

    #This saves the plot as a png to the protper filepath with a high dpi so the image is more clean and a bbox_inches to make sure the plot is not cut off with longer names.
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    #this stops plots from overlapping and closes the plot so the next one can be created without any issues.
    plt.close()

    #final print statment to let the user know that the plots have been saved to the output directory checpoint.
    print(f"Plot saved to {output_dir} folder!")
    
    

