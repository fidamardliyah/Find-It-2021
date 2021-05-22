TIM FOUND IT

# Load Dataset 


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
#Load Data From Local File
datatrain=pd.read_csv(r'D:/Coolyeah/LOMBA/Find It/public dataset/public-train.csv',sep='|')
datauji=pd.read_csv(r'D:/Coolyeah/LOMBA/Find It/public dataset/public-test.csv',sep='|')
```


```python
datatrain
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>dimension_1</th>
      <th>dimension_2</th>
      <th>genre_0</th>
      <th>genre_1</th>
      <th>genre_2</th>
      <th>genre_3</th>
      <th>genre_4</th>
      <th>genre_5</th>
      <th>genre_6</th>
      <th>genre_7</th>
      <th>genre_8</th>
      <th>genre_9</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>author2106</td>
      <td>Just after the Second World War, in the small ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>309.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.76</td>
      <td>NaN</td>
      <td>26625</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Historical Fiction</td>
      <td>Fiction</td>
      <td>Historical</td>
      <td>Audiobook</td>
      <td>Romance</td>
      <td>Books About Books</td>
      <td>Adult</td>
      <td>Adult Fiction</td>
      <td>British Literature</td>
      <td>Chick Lit</td>
      <td>0.45</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>author1018</td>
      <td>Blame it on Hawaii’s rainbows, sparkling beach...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.48</td>
      <td>NaN</td>
      <td>21</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Memoir</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>author1087</td>
      <td>The Pulitzer Prize–winning, bestselling author...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.56</td>
      <td>NaN</td>
      <td>59885</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nonfiction</td>
      <td>History</td>
      <td>Politics</td>
      <td>Race</td>
      <td>Social Justice</td>
      <td>Audiobook</td>
      <td>Sociology</td>
      <td>Anti Racist</td>
      <td>American History</td>
      <td>African American</td>
      <td>0.42</td>
      <td>0.22</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>author1295</td>
      <td>THINGS ARE ABOUT TO GET SERIOUS FOR HARRY DRES...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>418.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.39</td>
      <td>NaN</td>
      <td>26643</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fantasy</td>
      <td>Urban Fantasy</td>
      <td>Fiction</td>
      <td>Magic</td>
      <td>Paranormal</td>
      <td>Audiobook</td>
      <td>Vampires</td>
      <td>Mystery</td>
      <td>Supernatural</td>
      <td>Fae</td>
      <td>0.41</td>
      <td>0.30</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>author2622</td>
      <td>The Romanovs were the most successful dynasty ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>784.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.93</td>
      <td>NaN</td>
      <td>11772</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>History</td>
      <td>Nonfiction</td>
      <td>Russia</td>
      <td>Biography</td>
      <td>Historical</td>
      <td>Russian History</td>
      <td>Audiobook</td>
      <td>Politics</td>
      <td>European History</td>
      <td>Romanovs</td>
      <td>0.42</td>
      <td>0.30</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3545</th>
      <td>author1144</td>
      <td>How much is too much to love? Travis Maddox le...</td>
      <td>Paperback</td>
      <td>Original Edition</td>
      <td>448.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.19</td>
      <td>NaN</td>
      <td>172198</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Romance</td>
      <td>New Adult</td>
      <td>Contemporary</td>
      <td>Young Adult</td>
      <td>Contemporary Romance</td>
      <td>College</td>
      <td>Fiction</td>
      <td>Chick Lit</td>
      <td>Fighters</td>
      <td>Love</td>
      <td>0.36</td>
      <td>0.24</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.07</td>
      <td>0.06</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3546</th>
      <td>author2852</td>
      <td>Magneto and Professor X. Superman and Lex Luth...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>478.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.21</td>
      <td>NaN</td>
      <td>43149</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fantasy</td>
      <td>Science Fiction</td>
      <td>Adult</td>
      <td>Fiction</td>
      <td>Urban Fantasy</td>
      <td>Paranormal</td>
      <td>Superheroes</td>
      <td>Audiobook</td>
      <td>Adult Fiction</td>
      <td>Young Adult</td>
      <td>0.43</td>
      <td>0.18</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3547</th>
      <td>author1309</td>
      <td>Following the launch of her #1 New York Times ...</td>
      <td>Hardcover</td>
      <td>First Edition</td>
      <td>352.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.55</td>
      <td>NaN</td>
      <td>5811</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cookbooks</td>
      <td>Cooking</td>
      <td>Nonfiction</td>
      <td>Food</td>
      <td>Foodie</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.68</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3548</th>
      <td>author1816</td>
      <td>Bachelors, beware. For those who keep secrets ...</td>
      <td>Kindle Edition</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.64</td>
      <td>NaN</td>
      <td>14</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Historical Romance</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3549</th>
      <td>author0882</td>
      <td>In the thrilling, nerve-wracking finale of Eze...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>315.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.68</td>
      <td>NaN</td>
      <td>1959</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Horror</td>
      <td>Fiction</td>
      <td>Thriller</td>
      <td>Science Fiction</td>
      <td>Audiobook</td>
      <td>Apocalyptic</td>
      <td>Adult</td>
      <td>Survival</td>
      <td>Mystery Thriller</td>
      <td>Action</td>
      <td>0.58</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3550 rows × 39 columns</p>
</div>




```python
datauji.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>dimension_1</th>
      <th>dimension_2</th>
      <th>genre_0</th>
      <th>genre_1</th>
      <th>genre_2</th>
      <th>genre_3</th>
      <th>genre_4</th>
      <th>genre_5</th>
      <th>genre_6</th>
      <th>genre_7</th>
      <th>genre_8</th>
      <th>genre_9</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>author2305</td>
      <td>Rachel Friedman has always been the consummate...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>295.0</td>
      <td>March 29, 2011</td>
      <td>publisher034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3628.74</td>
      <td>3.79</td>
      <td>4.4</td>
      <td>3412</td>
      <td>178</td>
      <td>13.21</td>
      <td>1.78</td>
      <td>20.32</td>
      <td>Travel</td>
      <td>Nonfiction</td>
      <td>Memoir</td>
      <td>Adventure</td>
      <td>Biography</td>
      <td>Biography Memoir</td>
      <td>Chick Lit</td>
      <td>Adult</td>
      <td>Autobiography</td>
      <td>Contemporary</td>
      <td>0.53</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>129789.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>author0204</td>
      <td>As Dr. Marina Singh embarks upon an uncertain ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>353.0</td>
      <td>June 7, 2011</td>
      <td>publisher155</td>
      <td>NaN</td>
      <td>990</td>
      <td>NaN</td>
      <td>521.63</td>
      <td>3.88</td>
      <td>4.1</td>
      <td>168718</td>
      <td>2908</td>
      <td>15.24</td>
      <td>2.97</td>
      <td>22.86</td>
      <td>Fiction</td>
      <td>Contemporary</td>
      <td>Book Club</td>
      <td>Literary Fiction</td>
      <td>Adult Fiction</td>
      <td>Adult</td>
      <td>Mystery</td>
      <td>Audiobook</td>
      <td>Adventure</td>
      <td>Novels</td>
      <td>0.62</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>262465.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>author2300</td>
      <td>From the moment she took a job on Captain Cald...</td>
      <td>Paperback</td>
      <td>US edition</td>
      <td>373.0</td>
      <td>April 22, 2014</td>
      <td>publisher261</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5443.10</td>
      <td>3.96</td>
      <td>4.5</td>
      <td>6845</td>
      <td>304</td>
      <td>13.97</td>
      <td>2.54</td>
      <td>20.95</td>
      <td>Science Fiction</td>
      <td>Romance</td>
      <td>Space Opera</td>
      <td>Aliens</td>
      <td>Fiction</td>
      <td>Space</td>
      <td>Fantasy</td>
      <td>Adult</td>
      <td>Adventure</td>
      <td>Military Fiction</td>
      <td>0.44</td>
      <td>0.16</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>182195.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>author1746</td>
      <td>#1 New York Times bestseller Lisa Gardner, aut...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>423.0</td>
      <td>February 5, 2013</td>
      <td>publisher105</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>680.39</td>
      <td>4.07</td>
      <td>4.6</td>
      <td>30037</td>
      <td>1887</td>
      <td>16.51</td>
      <td>3.68</td>
      <td>23.75</td>
      <td>Mystery</td>
      <td>Thriller</td>
      <td>Suspense</td>
      <td>Fiction</td>
      <td>Mystery Thriller</td>
      <td>Crime</td>
      <td>Audiobook</td>
      <td>Adult Fiction</td>
      <td>Adult</td>
      <td>Detective</td>
      <td>0.30</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>288596.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>author1716</td>
      <td>This is not your mother’s memoir. In The Chron...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>310.0</td>
      <td>April 1, 2011</td>
      <td>publisher166</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>453.59</td>
      <td>4.23</td>
      <td>4.4</td>
      <td>9193</td>
      <td>463</td>
      <td>14.22</td>
      <td>2.03</td>
      <td>22.61</td>
      <td>Memoir</td>
      <td>Nonfiction</td>
      <td>Feminism</td>
      <td>Biography</td>
      <td>Queer</td>
      <td>LGBT</td>
      <td>Biography Memoir</td>
      <td>Autobiography</td>
      <td>Womens</td>
      <td>Writing</td>
      <td>0.44</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>230270.0</td>
    </tr>
  </tbody>
</table>
</div>



# Data Understanding

Mengetahui karakteristik data


```python
datatrain.isnull().sum()
```




    author_id           10
    description         38
    bookformat          13
    bookedition       3319
    pages               99
    published_date    2982
    publisher_id      2982
    reading_age       3424
    lexile_measure    3462
    grade_level       3450
    weight            3031
    rating_value_0      10
    rating_value_1    2997
    rating_count_0       0
    rating_count_1       0
    dimension_0       3038
    dimension_1       3038
    dimension_2       3051
    genre_0            150
    genre_1            197
    genre_2            228
    genre_3            248
    genre_4            280
    genre_5            310
    genre_6            338
    genre_7            378
    genre_8            414
    genre_9            450
    genre_0_weight     150
    genre_1_weight     197
    genre_2_weight     228
    genre_3_weight     248
    genre_4_weight     280
    genre_5_weight     310
    genre_6_weight     338
    genre_7_weight     378
    genre_8_weight     414
    genre_9_weight     450
    price             3007
    dtype: int64




```python
data= datatrain.dropna(subset=["author_id"], axis=0)
data.isnull().sum()
```




    author_id            0
    description         28
    bookformat           3
    bookedition       3309
    pages               89
    published_date    2972
    publisher_id      2972
    reading_age       3414
    lexile_measure    3452
    grade_level       3440
    weight            3021
    rating_value_0       0
    rating_value_1    2987
    rating_count_0       0
    rating_count_1       0
    dimension_0       3028
    dimension_1       3028
    dimension_2       3041
    genre_0            140
    genre_1            187
    genre_2            218
    genre_3            238
    genre_4            270
    genre_5            300
    genre_6            328
    genre_7            368
    genre_8            404
    genre_9            440
    genre_0_weight     140
    genre_1_weight     187
    genre_2_weight     218
    genre_3_weight     238
    genre_4_weight     270
    genre_5_weight     300
    genre_6_weight     328
    genre_7_weight     368
    genre_8_weight     404
    genre_9_weight     440
    price             2997
    dtype: int64




```python
print("datatrain:",data.shape)
```

    datatrain: (3540, 39)
    


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3540 entries, 0 to 3549
    Data columns (total 39 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   author_id       3540 non-null   object 
     1   description     3512 non-null   object 
     2   bookformat      3537 non-null   object 
     3   bookedition     231 non-null    object 
     4   pages           3451 non-null   float64
     5   published_date  568 non-null    object 
     6   publisher_id    568 non-null    object 
     7   reading_age     126 non-null    object 
     8   lexile_measure  88 non-null     object 
     9   grade_level     100 non-null    object 
     10  weight          519 non-null    float64
     11  rating_value_0  3540 non-null   float64
     12  rating_value_1  553 non-null    float64
     13  rating_count_0  3540 non-null   int64  
     14  rating_count_1  3540 non-null   int64  
     15  dimension_0     512 non-null    float64
     16  dimension_1     512 non-null    float64
     17  dimension_2     499 non-null    float64
     18  genre_0         3400 non-null   object 
     19  genre_1         3353 non-null   object 
     20  genre_2         3322 non-null   object 
     21  genre_3         3302 non-null   object 
     22  genre_4         3270 non-null   object 
     23  genre_5         3240 non-null   object 
     24  genre_6         3212 non-null   object 
     25  genre_7         3172 non-null   object 
     26  genre_8         3136 non-null   object 
     27  genre_9         3100 non-null   object 
     28  genre_0_weight  3400 non-null   float64
     29  genre_1_weight  3353 non-null   float64
     30  genre_2_weight  3322 non-null   float64
     31  genre_3_weight  3302 non-null   float64
     32  genre_4_weight  3270 non-null   float64
     33  genre_5_weight  3240 non-null   float64
     34  genre_6_weight  3212 non-null   float64
     35  genre_7_weight  3172 non-null   float64
     36  genre_8_weight  3136 non-null   float64
     37  genre_9_weight  3100 non-null   float64
     38  price           543 non-null    float64
    dtypes: float64(18), int64(2), object(19)
    memory usage: 1.1+ MB
    


```python
data.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pages</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>dimension_1</th>
      <th>dimension_2</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3451.000000</td>
      <td>519.000000</td>
      <td>3540.000000</td>
      <td>553.000000</td>
      <td>3.540000e+03</td>
      <td>3540.000000</td>
      <td>512.000000</td>
      <td>512.000000</td>
      <td>499.000000</td>
      <td>3400.000000</td>
      <td>3353.000000</td>
      <td>3322.000000</td>
      <td>3302.000000</td>
      <td>3270.000000</td>
      <td>3240.00000</td>
      <td>3212.000000</td>
      <td>3172.000000</td>
      <td>3136.000000</td>
      <td>3100.000000</td>
      <td>543.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>331.894234</td>
      <td>2372.200829</td>
      <td>4.025017</td>
      <td>4.519530</td>
      <td>5.520089e+04</td>
      <td>371.803672</td>
      <td>15.438125</td>
      <td>3.686113</td>
      <td>22.091182</td>
      <td>0.427418</td>
      <td>0.198375</td>
      <td>0.119124</td>
      <td>0.077935</td>
      <td>0.054976</td>
      <td>0.04166</td>
      <td>0.032699</td>
      <td>0.026434</td>
      <td>0.021875</td>
      <td>0.018174</td>
      <td>231296.762431</td>
    </tr>
    <tr>
      <th>std</th>
      <td>147.472358</td>
      <td>2232.405524</td>
      <td>0.560412</td>
      <td>0.284086</td>
      <td>1.838874e+05</td>
      <td>2049.434885</td>
      <td>3.438329</td>
      <td>4.004741</td>
      <td>3.236897</td>
      <td>0.165297</td>
      <td>0.070470</td>
      <td>0.046482</td>
      <td>0.031668</td>
      <td>0.022640</td>
      <td>0.01698</td>
      <td>0.013457</td>
      <td>0.011010</td>
      <td>0.009474</td>
      <td>0.008107</td>
      <td>138233.508496</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>400.070000</td>
      <td>0.000000</td>
      <td>3.200000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>1.470000</td>
      <td>0.030000</td>
      <td>1.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>256.000000</td>
      <td>612.350000</td>
      <td>3.880000</td>
      <td>4.400000</td>
      <td>3.639000e+03</td>
      <td>1.000000</td>
      <td>13.970000</td>
      <td>2.125000</td>
      <td>20.950000</td>
      <td>0.310000</td>
      <td>0.150000</td>
      <td>0.090000</td>
      <td>0.060000</td>
      <td>0.040000</td>
      <td>0.03000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>128922.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>330.000000</td>
      <td>839.150000</td>
      <td>4.080000</td>
      <td>4.600000</td>
      <td>1.320000e+04</td>
      <td>1.000000</td>
      <td>15.240000</td>
      <td>2.790000</td>
      <td>22.860000</td>
      <td>0.390000</td>
      <td>0.200000</td>
      <td>0.120000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.04000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>212946.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>400.000000</td>
      <td>4036.970000</td>
      <td>4.260000</td>
      <td>4.700000</td>
      <td>4.276075e+04</td>
      <td>1.000000</td>
      <td>16.360000</td>
      <td>3.580000</td>
      <td>24.130000</td>
      <td>0.500000</td>
      <td>0.240000</td>
      <td>0.150000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.05000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>287224.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1248.000000</td>
      <td>7212.110000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>3.803071e+06</td>
      <td>40409.000000</td>
      <td>30.480000</td>
      <td>27.430000</td>
      <td>28.910000</td>
      <td>1.000000</td>
      <td>0.600000</td>
      <td>0.330000</td>
      <td>0.250000</td>
      <td>0.150000</td>
      <td>0.12000</td>
      <td>0.110000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>978395.000000</td>
    </tr>
  </tbody>
</table>
</div>



# EDA

Analisis eksplorasi data


```python
import seaborn as sns

sns.countplot(data['price'])
```

    /usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      FutureWarning
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f69323b7b50>




![png](output_16_2.png)



```python
plt.figure(figsize=(10,5))
data['bookformat'].value_counts().plot.bar()
plt.title("Book Format")
plt.show()
```


![png](output_17_0.png)



```python
plt.figure(figsize=(15,5))
data['bookedition'].value_counts().plot.bar()
plt.title("Book Edition")
plt.show()
```


![png](output_18_0.png)



```python
plt.figure(figsize=(10,5))
data['reading_age'].value_counts().plot.bar()
plt.title("Reading Age")
plt.show()
```


![png](output_19_0.png)



```python
plt.figure(figsize=(10,5))
data['lexile_measure'].value_counts().plot.bar()
plt.title("Lexile Measure")
plt.show()
```


![png](output_20_0.png)



```python
plt.figure(figsize=(10,5))
data['grade_level'].value_counts().plot.bar()
plt.title("Grade Level")
plt.show()
```


![png](output_21_0.png)



```python
plt.figure(figsize=(10,5))
data['genre_0'].value_counts().plot.bar()
plt.title("Genre_0")
plt.show()
```


![png](output_22_0.png)



```python
fig, axes = plt.subplots(nrows=5, ncols=4 , figsize=(15,10))
axes[0,0].set_title("pages")
axes[0,1].set_title("weight")
axes[0,2].set_title("rating_value_0")
axes[0,3].set_title("rating_value_1")
axes[1,0].set_title("rating_count_0")
axes[1,1].set_title("rating_count_1")
axes[1,2].set_title("dimension_0")
axes[1,3].set_title("dimension_1")
axes[2,0].set_title("dimension_2")
axes[2,1].set_title("genre_0_weight")
axes[2,2].set_title("genre_1_weight")
axes[2,3].set_title("genre_2_weight")
axes[3,0].set_title("genre_3_weight")
axes[3,1].set_title("genre_4_weight")
axes[3,2].set_title("genre_5_weight")
axes[3,3].set_title("genre_6_weight")
axes[4,0].set_title("genre_7_weight")
axes[4,1].set_title("genre_8_weight")
axes[4,2].set_title("genre_9_weight")
axes[4,3].set_title("price")

sns.distplot(data["pages"], ax=axes[0,0])
sns.distplot(data["weight"], ax=axes[0,1])
sns.distplot(data["rating_value_0"], ax=axes[0,2])
sns.distplot(data["rating_value_1"], ax=axes[0,3])
sns.distplot(data["rating_count_0"], ax=axes[1,0])
sns.distplot(data["rating_count_1"], ax=axes[1,1])
sns.distplot(data["dimension_0"], ax=axes[1,2])
sns.distplot(data["dimension_1"], ax=axes[1,3])
sns.distplot(data["dimension_2"], ax=axes[2,0])
sns.distplot(data["genre_0_weight"], ax=axes[2,1])
sns.distplot(data["genre_1_weight"], ax=axes[2,2])
sns.distplot(data["genre_2_weight"], ax=axes[2,3])
sns.distplot(data["genre_3_weight"], ax=axes[3,0])
sns.distplot(data["genre_4_weight"], ax=axes[3,1])
sns.distplot(data["genre_5_weight"], ax=axes[3,2])
sns.distplot(data["genre_6_weight"], ax=axes[3,3])
sns.distplot(data["genre_7_weight"], ax=axes[4,0])
sns.distplot(data["genre_8_weight"], ax=axes[4,1])
sns.distplot(data["genre_9_weight"], ax=axes[4,2])
sns.distplot(data["price"], ax=axes[4,3])

fig.tight_layout()
```

    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    


![png](output_23_1.png)



```python
plt.figure(figsize=(10,5))
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")
```


```python
data['bookedition'].value_counts()
```




    First Edition           76
    1st Edition             20
    Trade                   13
    First Edition (U.S.)    11
    1st                      9
                            ..
    Simon Pulse              1
    Yen                      1
    Hardcover                1
    Original Edition         1
    second edition           1
    Name: bookedition, Length: 68, dtype: int64




```python
data['reading_age'].value_counts()
```




    8 - 12 years       14
    18 years and up    14
    14 years and up    12
    4 - 8 years        11
    13 years and up    10
    10 - 14 years       7
    12 - 15 years       7
    13 - 17 years       7
    14 - 17 years       6
    15 years and up     4
    12 - 17 years       4
    16 years and up     3
    12 - 18 years       3
    3 - 5 years         2
    3 - 7 years         2
    10 - 13 years       2
    2 - 5 years         2
    12 years and up     2
    11 years and up     2
    4 - 7 years         1
    6 - 11 years        1
    3 - 8 years         1
    8 - 11 years        1
    5 - 8 years         1
    10 - 12 years       1
    4 - 6 years         1
    13 - 18 years       1
    9 - 12 years        1
    3 - 6 years         1
    10 years and up     1
    10 years            1
    Name: reading_age, dtype: int64




```python
fig, ax = plt.subplots(figsize=(10,8))
corr = data.corr()
sns.heatmap(corr)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f69314d56d0>




![png](output_27_1.png)


# Preprocessing Data

Hapus Duplicate Data


```python
data.duplicated().sum()
```




    100




```python
find = data.drop_duplicates()
```


Hapus Variabel Tidak Dibutuhkan





```python
findit = find.drop(labels=['author_id','description','publisher_id', 'published_date','bookedition'], axis=1)
```

## Imputasi 


```python
#Melihat kemiringan dari data untuk imputasi
findit.skew(axis=0, skipna=True)
```




    pages              0.771851
    weight             0.820661
    rating_value_0    -4.962321
    rating_value_1    -1.100762
    rating_count_0    12.041248
    rating_count_1    11.133942
    dimension_0        0.373369
    dimension_1        3.690633
    dimension_2       -2.867969
    genre_0_weight     1.454005
    genre_1_weight     0.129424
    genre_2_weight     0.247111
    genre_3_weight     0.419861
    genre_4_weight     0.391143
    genre_5_weight     0.358376
    genre_6_weight     0.491903
    genre_7_weight     0.405005
    genre_8_weight     0.446851
    genre_9_weight     0.518299
    price              1.400949
    dtype: float64




```python
#imputasi numerik
findit['pages'].fillna(findit['pages'].mean(),inplace=True)
findit['weight'].fillna(findit['weight'].mean(),inplace=True)
findit['rating_value_0'].fillna(findit['rating_value_0'].median(),inplace=True)
findit['rating_value_1'].fillna(findit['rating_value_1'].mean(),inplace=True)
findit['rating_count_0'].fillna(findit['rating_count_0'].median(),inplace=True)
findit['rating_count_1'].fillna(findit['rating_count_1'].median(),inplace=True)
findit['dimension_0'].fillna(findit['dimension_0'].mean(),inplace=True)
findit['dimension_1'].fillna(findit['dimension_1'].median(),inplace=True)
findit['dimension_2'].fillna(findit['dimension_2'].median(),inplace=True)
findit['genre_0_weight'].fillna(findit['genre_0_weight'].mean(),inplace=True)
findit['genre_1_weight'].fillna(findit['genre_1_weight'].mean(),inplace=True)
findit['genre_2_weight'].fillna(findit['genre_2_weight'].mean(),inplace=True)
findit['genre_3_weight'].fillna(findit['genre_3_weight'].mean(),inplace=True)
findit['genre_4_weight'].fillna(findit['genre_4_weight'].mean(),inplace=True)
findit['genre_5_weight'].fillna(findit['genre_5_weight'].mean(),inplace=True)
findit['genre_6_weight'].fillna(findit['genre_6_weight'].mean(),inplace=True)
findit['genre_7_weight'].fillna(findit['genre_7_weight'].mean(),inplace=True)
findit['genre_8_weight'].fillna(findit['genre_8_weight'].mean(),inplace=True)
findit['genre_9_weight'].fillna(findit['genre_9_weight'].mean(),inplace=True)
```


```python
#imputasi string
findit['bookformat'].fillna(("Hardcover"),inplace=True)
findit['genre_0'].fillna("Historical Fiction",inplace=True)
findit['genre_1'].fillna("Fiction",inplace=True)
findit['genre_2'].fillna("Historical",inplace=True)
findit['genre_3'].fillna("Audiobook",inplace=True)
findit['genre_4'].fillna("Romance",inplace=True)
findit['genre_5'].fillna("Books About Books",inplace=True)
findit['genre_6'].fillna("Adult",inplace=True)
findit['genre_7'].fillna("Adult Fiction",inplace=True)
findit['genre_8'].fillna("British Literature",inplace=True)
findit['genre_9'].fillna("Chick Lit",inplace=True)
findit['reading_age'].fillna("18 years and up", inplace=True)
findit['lexile_measure'].fillna("710L", inplace=True)
findit['grade_level'].fillna("7 - 9", inplace=True)
```


```python
mis1 = findit.isnull().sum()
print(mis1)
```

    bookformat           0
    pages                0
    reading_age          0
    lexile_measure       0
    grade_level          0
    weight               0
    rating_value_0       0
    rating_value_1       0
    rating_count_0       0
    rating_count_1       0
    dimension_0          0
    dimension_1          0
    dimension_2          0
    genre_0              0
    genre_1              0
    genre_2              0
    genre_3              0
    genre_4              0
    genre_5              0
    genre_6              0
    genre_7              0
    genre_8              0
    genre_9              0
    genre_0_weight       0
    genre_1_weight       0
    genre_2_weight       0
    genre_3_weight       0
    genre_4_weight       0
    genre_5_weight       0
    genre_6_weight       0
    genre_7_weight       0
    genre_8_weight       0
    genre_9_weight       0
    price             2904
    dtype: int64
    

 # Encoding Variables


```python
#Separating categorical and numerical columns
num_cols   = ['pages', 'weight', 'rating_value_0','rating_value_1','rating_count_0','rating_count_1','genre_0_weight','genre_1_weight','genre_2_weight','genre_3_weight','genre_4_weight','genre_5_weight','genre_6_weight','genre_7_weight','genre_8_weight','genre_9_weight','price']

#multi category columns
multi_cols = ['bookformat','genre_0','genre_1','genre_2','genre_3','genre_4','genre_5','genre_6','genre_7','genre_8','genre_9']

#ordinal
ordi_cols = ['lexile_measure','reading_age','grade_level']
```


```python
train = findit
```


```python
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
train[['lexile_measure']] = ord_enc.fit_transform(train[["lexile_measure"]])
train[['reading_age']] = ord_enc.fit_transform(train[["reading_age"]])
train[['grade_level']] = ord_enc.fit_transform(train[["grade_level"]])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Hardcover</td>
      <td>309.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.76</td>
      <td>4.518899</td>
      <td>26625</td>
      <td>1</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Paperback</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.48</td>
      <td>4.518899</td>
      <td>21</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Hardcover</td>
      <td>496.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.56</td>
      <td>4.518899</td>
      <td>59885</td>
      <td>1</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Hardcover</td>
      <td>418.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.39</td>
      <td>4.518899</td>
      <td>26643</td>
      <td>1</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Hardcover</td>
      <td>784.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.93</td>
      <td>4.518899</td>
      <td>11772</td>
      <td>1</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
train['bookformat'] = labelencoder.fit_transform(train['bookformat'])
train['genre_0'] = labelencoder.fit_transform(train['genre_0'])
train['genre_1'] = labelencoder.fit_transform(train['genre_1'])
train['genre_2'] = labelencoder.fit_transform(train['genre_2'])
train['genre_3'] = labelencoder.fit_transform(train['genre_3'])
train['genre_4'] = labelencoder.fit_transform(train['genre_4'])
train['genre_5'] = labelencoder.fit_transform(train['genre_5'])
train['genre_6'] = labelencoder.fit_transform(train['genre_6'])
train['genre_7'] = labelencoder.fit_transform(train['genre_7'])
train['genre_8'] = labelencoder.fit_transform(train['genre_8'])
train['genre_9'] = labelencoder.fit_transform(train['genre_9'])
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>309.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.76</td>
      <td>4.518899</td>
      <td>26625</td>
      <td>1</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>7</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.48</td>
      <td>4.518899</td>
      <td>21</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>496.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.56</td>
      <td>4.518899</td>
      <td>59885</td>
      <td>1</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>418.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.39</td>
      <td>4.518899</td>
      <td>26643</td>
      <td>1</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>784.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.93</td>
      <td>4.518899</td>
      <td>11772</td>
      <td>1</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3545</td>
      <td>7</td>
      <td>448.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.19</td>
      <td>4.518899</td>
      <td>172198</td>
      <td>1</td>
      <td>...</td>
      <td>0.240000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3546</td>
      <td>2</td>
      <td>478.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.21</td>
      <td>4.518899</td>
      <td>43149</td>
      <td>1</td>
      <td>...</td>
      <td>0.180000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3547</td>
      <td>2</td>
      <td>352.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.55</td>
      <td>4.518899</td>
      <td>5811</td>
      <td>1</td>
      <td>...</td>
      <td>0.140000</td>
      <td>0.130000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3548</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.64</td>
      <td>4.518899</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3549</td>
      <td>2</td>
      <td>315.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.68</td>
      <td>4.518899</td>
      <td>1959</td>
      <td>1</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.090000</td>
      <td>0.090000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3440 rows × 34 columns</p>
</div>



# Data Partition

Memisahkan Data Label dan Unlabeled


```python
label= train.dropna(subset=["price"], axis=0)
label['price']
```




    9        98172.0
    17       57604.0
    29      103658.0
    33      649665.0
    44      247883.0
              ...   
    3518    262176.0
    3522    216411.0
    3529    152310.0
    3538    176853.0
    3541    216555.0
    Name: price, Length: 536, dtype: float64




```python
unlabel= train[train['price'].isnull()]
unlabel['price']
```




    0      NaN
    1      NaN
    2      NaN
    3      NaN
    4      NaN
            ..
    3545   NaN
    3546   NaN
    3547   NaN
    3548   NaN
    3549   NaN
    Name: price, Length: 2904, dtype: float64




```python
##seperating dependent and independent variables 
label_X = label.drop(labels='price',axis=1)
label_Y = label['price']
```


```python
label_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9</td>
      <td>5</td>
      <td>504.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3628.740000</td>
      <td>4.29</td>
      <td>4.600000</td>
      <td>26983</td>
      <td>504</td>
      <td>...</td>
      <td>0.260000</td>
      <td>0.210000</td>
      <td>0.170000</td>
      <td>0.100000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>17</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.89</td>
      <td>4.518899</td>
      <td>27</td>
      <td>1</td>
      <td>...</td>
      <td>0.428579</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
    <tr>
      <td>29</td>
      <td>7</td>
      <td>324.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3991.610000</td>
      <td>3.99</td>
      <td>4.600000</td>
      <td>43657</td>
      <td>1537</td>
      <td>...</td>
      <td>0.450000</td>
      <td>0.190000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>33</td>
      <td>2</td>
      <td>528.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>771.110000</td>
      <td>3.68</td>
      <td>4.300000</td>
      <td>19382</td>
      <td>1504</td>
      <td>...</td>
      <td>0.770000</td>
      <td>0.070000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <td>44</td>
      <td>7</td>
      <td>500.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>725.750000</td>
      <td>3.81</td>
      <td>3.600000</td>
      <td>32</td>
      <td>9</td>
      <td>...</td>
      <td>0.428579</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
unlabel_X = unlabel.drop(labels='price',axis=1)
unlabel_Y = unlabel['price']
```


```python
unlabel_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>309.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.76</td>
      <td>4.518899</td>
      <td>26625</td>
      <td>1</td>
      <td>...</td>
      <td>0.45</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>7</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.48</td>
      <td>4.518899</td>
      <td>21</td>
      <td>1</td>
      <td>...</td>
      <td>1.00</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>496.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.56</td>
      <td>4.518899</td>
      <td>59885</td>
      <td>1</td>
      <td>...</td>
      <td>0.42</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>418.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.39</td>
      <td>4.518899</td>
      <td>26643</td>
      <td>1</td>
      <td>...</td>
      <td>0.41</td>
      <td>0.300000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>784.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.93</td>
      <td>4.518899</td>
      <td>11772</td>
      <td>1</td>
      <td>...</td>
      <td>0.42</td>
      <td>0.300000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



#First Model with Label Data


## Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=123)
param_grid = { 
    'n_estimators': [200,500,1000, 1500],
    'max_features': ['auto','log2'],
    'criterion' :['entropy','gini']
}
```


```python
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(label_X, label_Y)
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:668: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      % (min_groups, self.n_splits)), UserWarning)
    




    GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=123),
                 param_grid={'criterion': ['entropy', 'gini'],
                             'max_features': ['auto', 'log2'],
                             'n_estimators': [200, 500, 1000, 1500]})




```python
CV_rfc.best_params_
```




    {'criterion': 'gini', 'max_features': 'auto', 'n_estimators': 200}




```python
pred=CV_rfc.predict(label_X)
```


```python
from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ",accuracy_score(label_Y,pred))
```

    Accuracy for Random Forest on CV data:  0.9925373134328358
    


```python
RMSE = np.sqrt(np.mean(pow(pred - label_Y, 2)))
RMSE
```




    6207.430100411324




```python
hasilpred=pd.DataFrame(pred)
hasilpred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>98172.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>57604.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>103658.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>649665.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>247883.0</td>
    </tr>
  </tbody>
</table>
</div>



# Prediksi Data Unlabeled


```python
unlabel_pred= CV_rfc.predict(unlabel_X)
hasilunlabel = pd.DataFrame(unlabel_pred)
hasilunlabel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>43167.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>72041.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>404236.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2899</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>144226.0</td>
    </tr>
    <tr>
      <td>2901</td>
      <td>389655.0</td>
    </tr>
    <tr>
      <td>2902</td>
      <td>144226.0</td>
    </tr>
    <tr>
      <td>2903</td>
      <td>324688.0</td>
    </tr>
  </tbody>
</table>
<p>2904 rows × 1 columns</p>
</div>



# Combining Label Data and Unlabel Data


```python
df2= pd.DataFrame(unlabel)
df2.reset_index(level=0, inplace=True)
df2['price']= hasilunlabel
df2 = df2.drop('index', axis=1)
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>309.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.76</td>
      <td>4.518899</td>
      <td>26625</td>
      <td>1</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>43167.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>7</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.48</td>
      <td>4.518899</td>
      <td>21</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>72041.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>496.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.56</td>
      <td>4.518899</td>
      <td>59885</td>
      <td>1</td>
      <td>...</td>
      <td>0.220000</td>
      <td>0.080000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>418.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.39</td>
      <td>4.518899</td>
      <td>26643</td>
      <td>1</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>784.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.93</td>
      <td>4.518899</td>
      <td>11772</td>
      <td>1</td>
      <td>...</td>
      <td>0.300000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>404236.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2899</td>
      <td>7</td>
      <td>448.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.19</td>
      <td>4.518899</td>
      <td>172198</td>
      <td>1</td>
      <td>...</td>
      <td>0.240000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>2900</td>
      <td>2</td>
      <td>478.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.21</td>
      <td>4.518899</td>
      <td>43149</td>
      <td>1</td>
      <td>...</td>
      <td>0.180000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>144226.0</td>
    </tr>
    <tr>
      <td>2901</td>
      <td>2</td>
      <td>352.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.55</td>
      <td>4.518899</td>
      <td>5811</td>
      <td>1</td>
      <td>...</td>
      <td>0.140000</td>
      <td>0.130000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>389655.0</td>
    </tr>
    <tr>
      <td>2902</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.64</td>
      <td>4.518899</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>144226.0</td>
    </tr>
    <tr>
      <td>2903</td>
      <td>2</td>
      <td>315.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.68</td>
      <td>4.518899</td>
      <td>1959</td>
      <td>1</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.090000</td>
      <td>0.090000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>324688.0</td>
    </tr>
  </tbody>
</table>
<p>2904 rows × 34 columns</p>
</div>




```python
df1 = label
df1.reset_index(level=0, inplace=True)
df1 = df1.drop('index', axis=1)
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5</td>
      <td>504.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3628.740000</td>
      <td>4.29</td>
      <td>4.600000</td>
      <td>26983</td>
      <td>504</td>
      <td>...</td>
      <td>0.210000</td>
      <td>0.170000</td>
      <td>0.100000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>98172.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.89</td>
      <td>4.518899</td>
      <td>27</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>57604.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7</td>
      <td>324.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3991.610000</td>
      <td>3.99</td>
      <td>4.600000</td>
      <td>43657</td>
      <td>1537</td>
      <td>...</td>
      <td>0.190000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>103658.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>528.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>771.110000</td>
      <td>3.68</td>
      <td>4.300000</td>
      <td>19382</td>
      <td>1504</td>
      <td>...</td>
      <td>0.070000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>649665.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>500.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>725.750000</td>
      <td>3.81</td>
      <td>3.600000</td>
      <td>32</td>
      <td>9</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>247883.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>531</td>
      <td>2</td>
      <td>351.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>512.560000</td>
      <td>3.83</td>
      <td>4.200000</td>
      <td>71693</td>
      <td>5943</td>
      <td>...</td>
      <td>0.200000</td>
      <td>0.170000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.070000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>262176.0</td>
    </tr>
    <tr>
      <td>532</td>
      <td>2</td>
      <td>257.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>566.990000</td>
      <td>4.12</td>
      <td>4.700000</td>
      <td>14157</td>
      <td>1407</td>
      <td>...</td>
      <td>0.250000</td>
      <td>0.160000</td>
      <td>0.060000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>216411.0</td>
    </tr>
    <tr>
      <td>533</td>
      <td>2</td>
      <td>444.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>780.180000</td>
      <td>3.84</td>
      <td>4.400000</td>
      <td>7758</td>
      <td>240</td>
      <td>...</td>
      <td>0.230000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>152310.0</td>
    </tr>
    <tr>
      <td>534</td>
      <td>2</td>
      <td>64.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3265.860000</td>
      <td>4.26</td>
      <td>4.800000</td>
      <td>85767</td>
      <td>15504</td>
      <td>...</td>
      <td>0.140000</td>
      <td>0.090000</td>
      <td>0.080000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>176853.0</td>
    </tr>
    <tr>
      <td>535</td>
      <td>7</td>
      <td>76.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>1741.790000</td>
      <td>4.24</td>
      <td>4.800000</td>
      <td>85</td>
      <td>11</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.080000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>216555.0</td>
    </tr>
  </tbody>
</table>
<p>536 rows × 34 columns</p>
</div>




```python
gabungan = pd.concat([df1,df2])
gabungan.reset_index(level=0, inplace=True)
gabungan = gabungan.drop('index', axis=1)
gabungan
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5</td>
      <td>504.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3628.740000</td>
      <td>4.29</td>
      <td>4.600000</td>
      <td>26983</td>
      <td>504</td>
      <td>...</td>
      <td>0.210000</td>
      <td>0.170000</td>
      <td>0.100000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>98172.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.89</td>
      <td>4.518899</td>
      <td>27</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>57604.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7</td>
      <td>324.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3991.610000</td>
      <td>3.99</td>
      <td>4.600000</td>
      <td>43657</td>
      <td>1537</td>
      <td>...</td>
      <td>0.190000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>103658.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>528.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>771.110000</td>
      <td>3.68</td>
      <td>4.300000</td>
      <td>19382</td>
      <td>1504</td>
      <td>...</td>
      <td>0.070000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>649665.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>500.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>725.750000</td>
      <td>3.81</td>
      <td>3.600000</td>
      <td>32</td>
      <td>9</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>247883.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3435</td>
      <td>7</td>
      <td>448.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.19</td>
      <td>4.518899</td>
      <td>172198</td>
      <td>1</td>
      <td>...</td>
      <td>0.240000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>3436</td>
      <td>2</td>
      <td>478.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.21</td>
      <td>4.518899</td>
      <td>43149</td>
      <td>1</td>
      <td>...</td>
      <td>0.180000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>144226.0</td>
    </tr>
    <tr>
      <td>3437</td>
      <td>2</td>
      <td>352.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.55</td>
      <td>4.518899</td>
      <td>5811</td>
      <td>1</td>
      <td>...</td>
      <td>0.140000</td>
      <td>0.130000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>389655.0</td>
    </tr>
    <tr>
      <td>3438</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.64</td>
      <td>4.518899</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
      <td>144226.0</td>
    </tr>
    <tr>
      <td>3439</td>
      <td>2</td>
      <td>315.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.68</td>
      <td>4.518899</td>
      <td>1959</td>
      <td>1</td>
      <td>...</td>
      <td>0.100000</td>
      <td>0.090000</td>
      <td>0.090000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>324688.0</td>
    </tr>
  </tbody>
</table>
<p>3440 rows × 34 columns</p>
</div>



#Second Model with combining data

# Model Predict


```python
gab_X = gabungan.drop(labels='price',axis=1)
gab_Y = gabungan['price']
```


```python
gab_X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5</td>
      <td>504.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3628.740000</td>
      <td>4.29</td>
      <td>4.600000</td>
      <td>26983</td>
      <td>504</td>
      <td>...</td>
      <td>0.260000</td>
      <td>0.210000</td>
      <td>0.170000</td>
      <td>0.100000</td>
      <td>0.080000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.89</td>
      <td>4.518899</td>
      <td>27</td>
      <td>1</td>
      <td>...</td>
      <td>0.428579</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7</td>
      <td>324.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>3991.610000</td>
      <td>3.99</td>
      <td>4.600000</td>
      <td>43657</td>
      <td>1537</td>
      <td>...</td>
      <td>0.450000</td>
      <td>0.190000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.050000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>528.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>771.110000</td>
      <td>3.68</td>
      <td>4.300000</td>
      <td>19382</td>
      <td>1504</td>
      <td>...</td>
      <td>0.770000</td>
      <td>0.070000</td>
      <td>0.050000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>500.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>725.750000</td>
      <td>3.81</td>
      <td>3.600000</td>
      <td>32</td>
      <td>9</td>
      <td>...</td>
      <td>0.428579</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3435</td>
      <td>7</td>
      <td>448.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.19</td>
      <td>4.518899</td>
      <td>172198</td>
      <td>1</td>
      <td>...</td>
      <td>0.360000</td>
      <td>0.240000</td>
      <td>0.100000</td>
      <td>0.070000</td>
      <td>0.070000</td>
      <td>0.060000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>3436</td>
      <td>2</td>
      <td>478.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.21</td>
      <td>4.518899</td>
      <td>43149</td>
      <td>1</td>
      <td>...</td>
      <td>0.430000</td>
      <td>0.180000</td>
      <td>0.110000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>3437</td>
      <td>2</td>
      <td>352.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.55</td>
      <td>4.518899</td>
      <td>5811</td>
      <td>1</td>
      <td>...</td>
      <td>0.680000</td>
      <td>0.140000</td>
      <td>0.130000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
    <tr>
      <td>3438</td>
      <td>3</td>
      <td>330.312146</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>4.64</td>
      <td>4.518899</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.198604</td>
      <td>0.119168</td>
      <td>0.077811</td>
      <td>0.054795</td>
      <td>0.041516</td>
      <td>0.032561</td>
      <td>0.026318</td>
      <td>0.021769</td>
      <td>0.018093</td>
    </tr>
    <tr>
      <td>3439</td>
      <td>2</td>
      <td>315.000000</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>2396.146133</td>
      <td>3.68</td>
      <td>4.518899</td>
      <td>1959</td>
      <td>1</td>
      <td>...</td>
      <td>0.580000</td>
      <td>0.100000</td>
      <td>0.090000</td>
      <td>0.090000</td>
      <td>0.040000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
    </tr>
  </tbody>
</table>
<p>3440 rows × 33 columns</p>
</div>




```python
rfc1=RandomForestClassifier(n_estimators=500)
rfc1.fit(gab_X,gab_Y)
predgab=rfc1.predict(gab_X)
```


```python
pred_gab=pd.DataFrame(pred)
pred_gab.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>98172.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>57604.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>103658.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>649665.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>247883.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
RMSEgab = np.sqrt(np.mean(pow(predgab - gab_Y, 2)))
RMSEgab
```




    2450.276029182436



# Preprocessing Data Uji


```python
datauji.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author_id</th>
      <th>description</th>
      <th>bookformat</th>
      <th>bookedition</th>
      <th>pages</th>
      <th>published_date</th>
      <th>publisher_id</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>author2305</td>
      <td>Rachel Friedman has always been the consummate...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>295.0</td>
      <td>March 29, 2011</td>
      <td>publisher034</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>129789.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>author0204</td>
      <td>As Dr. Marina Singh embarks upon an uncertain ...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>353.0</td>
      <td>June 7, 2011</td>
      <td>publisher155</td>
      <td>NaN</td>
      <td>990</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>262465.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>author2300</td>
      <td>From the moment she took a job on Captain Cald...</td>
      <td>Paperback</td>
      <td>US edition</td>
      <td>373.0</td>
      <td>April 22, 2014</td>
      <td>publisher261</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.16</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>182195.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>author1746</td>
      <td>#1 New York Times bestseller Lisa Gardner, aut...</td>
      <td>Hardcover</td>
      <td>NaN</td>
      <td>423.0</td>
      <td>February 5, 2013</td>
      <td>publisher105</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>288596.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>author1716</td>
      <td>This is not your mother’s memoir. In The Chron...</td>
      <td>Paperback</td>
      <td>NaN</td>
      <td>310.0</td>
      <td>April 1, 2011</td>
      <td>publisher166</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>230270.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
datauji.isnull().sum()
```




    author_id           0
    description         8
    bookformat          0
    bookedition       446
    pages              20
    published_date     31
    publisher_id       31
    reading_age       381
    lexile_measure    403
    grade_level       405
    weight             62
    rating_value_0      0
    rating_value_1     56
    rating_count_0      0
    rating_count_1      0
    dimension_0        63
    dimension_1        63
    dimension_2        81
    genre_0            41
    genre_1            47
    genre_2            55
    genre_3            63
    genre_4            68
    genre_5            73
    genre_6            77
    genre_7            80
    genre_8            82
    genre_9            87
    genre_0_weight     41
    genre_1_weight     47
    genre_2_weight     55
    genre_3_weight     63
    genre_4_weight     68
    genre_5_weight     73
    genre_6_weight     77
    genre_7_weight     80
    genre_8_weight     82
    genre_9_weight     87
    price               0
    dtype: int64




```python
datauji = datauji.drop(labels=['author_id','description','publisher_id', 'published_date','bookedition'], axis=1)
```


```python
#imputasi numerik
datauji['pages'].fillna(datauji['pages'].mean(),inplace=True)
datauji['weight'].fillna(datauji['weight'].mean(),inplace=True)
datauji['rating_value_0'].fillna(datauji['rating_value_0'].median(),inplace=True)
datauji['rating_value_1'].fillna(datauji['rating_value_1'].mean(),inplace=True)
datauji['rating_count_0'].fillna(datauji['rating_count_0'].median(),inplace=True)
datauji['rating_count_1'].fillna(datauji['rating_count_1'].median(),inplace=True)
datauji['dimension_0'].fillna(datauji['dimension_0'].mean(),inplace=True)
datauji['dimension_1'].fillna(datauji['dimension_1'].median(),inplace=True)
datauji['dimension_2'].fillna(datauji['dimension_2'].median(),inplace=True)
datauji['genre_0_weight'].fillna(datauji['genre_0_weight'].mean(),inplace=True)
datauji['genre_1_weight'].fillna(datauji['genre_1_weight'].mean(),inplace=True)
datauji['genre_2_weight'].fillna(datauji['genre_2_weight'].mean(),inplace=True)
datauji['genre_3_weight'].fillna(datauji['genre_3_weight'].mean(),inplace=True)
datauji['genre_4_weight'].fillna(datauji['genre_4_weight'].mean(),inplace=True)
datauji['genre_5_weight'].fillna(datauji['genre_5_weight'].mean(),inplace=True)
datauji['genre_6_weight'].fillna(datauji['genre_6_weight'].mean(),inplace=True)
datauji['genre_7_weight'].fillna(datauji['genre_7_weight'].mean(),inplace=True)
datauji['genre_8_weight'].fillna(datauji['genre_8_weight'].mean(),inplace=True)
datauji['genre_9_weight'].fillna(datauji['genre_9_weight'].mean(),inplace=True)
```


```python
#imputasi string
datauji['bookformat'].fillna(("Hardcover"),inplace=True)
datauji['genre_0'].fillna("Historical Fiction",inplace=True)
datauji['genre_1'].fillna("Fiction",inplace=True)
datauji['genre_2'].fillna("Historical",inplace=True)
datauji['genre_3'].fillna("Audiobook",inplace=True)
datauji['genre_4'].fillna("Romance",inplace=True)
datauji['genre_5'].fillna("Books About Books",inplace=True)
datauji['genre_6'].fillna("Adult",inplace=True)
datauji['genre_7'].fillna("Adult Fiction",inplace=True)
datauji['genre_8'].fillna("British Literature",inplace=True)
datauji['genre_9'].fillna("Chick Lit",inplace=True)
datauji['reading_age'].fillna("18 years and up", inplace=True)
datauji['lexile_measure'].fillna("710L", inplace=True)
datauji['grade_level'].fillna("7 - 9", inplace=True)
```


```python
datauji.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>dimension_0</th>
      <th>dimension_1</th>
      <th>dimension_2</th>
      <th>genre_0</th>
      <th>genre_1</th>
      <th>genre_2</th>
      <th>genre_3</th>
      <th>genre_4</th>
      <th>genre_5</th>
      <th>genre_6</th>
      <th>genre_7</th>
      <th>genre_8</th>
      <th>genre_9</th>
      <th>genre_0_weight</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paperback</td>
      <td>295.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3628.74</td>
      <td>3.79</td>
      <td>4.4</td>
      <td>3412</td>
      <td>178</td>
      <td>13.21</td>
      <td>1.78</td>
      <td>20.32</td>
      <td>Travel</td>
      <td>Nonfiction</td>
      <td>Memoir</td>
      <td>Adventure</td>
      <td>Biography</td>
      <td>Biography Memoir</td>
      <td>Chick Lit</td>
      <td>Adult</td>
      <td>Autobiography</td>
      <td>Contemporary</td>
      <td>0.53</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>129789.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hardcover</td>
      <td>353.0</td>
      <td>0</td>
      <td>990</td>
      <td>0</td>
      <td>521.63</td>
      <td>3.88</td>
      <td>4.1</td>
      <td>168718</td>
      <td>2908</td>
      <td>15.24</td>
      <td>2.97</td>
      <td>22.86</td>
      <td>Fiction</td>
      <td>Contemporary</td>
      <td>Book Club</td>
      <td>Literary Fiction</td>
      <td>Adult Fiction</td>
      <td>Adult</td>
      <td>Mystery</td>
      <td>Audiobook</td>
      <td>Adventure</td>
      <td>Novels</td>
      <td>0.62</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>262465.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paperback</td>
      <td>373.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5443.10</td>
      <td>3.96</td>
      <td>4.5</td>
      <td>6845</td>
      <td>304</td>
      <td>13.97</td>
      <td>2.54</td>
      <td>20.95</td>
      <td>Science Fiction</td>
      <td>Romance</td>
      <td>Space Opera</td>
      <td>Aliens</td>
      <td>Fiction</td>
      <td>Space</td>
      <td>Fantasy</td>
      <td>Adult</td>
      <td>Adventure</td>
      <td>Military Fiction</td>
      <td>0.44</td>
      <td>0.16</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>182195.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hardcover</td>
      <td>423.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>680.39</td>
      <td>4.07</td>
      <td>4.6</td>
      <td>30037</td>
      <td>1887</td>
      <td>16.51</td>
      <td>3.68</td>
      <td>23.75</td>
      <td>Mystery</td>
      <td>Thriller</td>
      <td>Suspense</td>
      <td>Fiction</td>
      <td>Mystery Thriller</td>
      <td>Crime</td>
      <td>Audiobook</td>
      <td>Adult Fiction</td>
      <td>Adult</td>
      <td>Detective</td>
      <td>0.30</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>288596.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Paperback</td>
      <td>310.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>453.59</td>
      <td>4.23</td>
      <td>4.4</td>
      <td>9193</td>
      <td>463</td>
      <td>14.22</td>
      <td>2.03</td>
      <td>22.61</td>
      <td>Memoir</td>
      <td>Nonfiction</td>
      <td>Feminism</td>
      <td>Biography</td>
      <td>Queer</td>
      <td>LGBT</td>
      <td>Biography Memoir</td>
      <td>Autobiography</td>
      <td>Womens</td>
      <td>Writing</td>
      <td>0.44</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>230270.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>Hardcover</td>
      <td>448.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>771.11</td>
      <td>4.15</td>
      <td>4.6</td>
      <td>7953</td>
      <td>409</td>
      <td>15.88</td>
      <td>4.45</td>
      <td>23.50</td>
      <td>History</td>
      <td>Nonfiction</td>
      <td>World War I</td>
      <td>War</td>
      <td>Military History</td>
      <td>Politics</td>
      <td>Military Fiction</td>
      <td>European History</td>
      <td>World History</td>
      <td>Historical</td>
      <td>0.50</td>
      <td>0.22</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>352263.0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>Hardcover</td>
      <td>589.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>748.43</td>
      <td>3.65</td>
      <td>3.8</td>
      <td>39510</td>
      <td>2920</td>
      <td>15.24</td>
      <td>3.73</td>
      <td>22.86</td>
      <td>Historical Fiction</td>
      <td>Fiction</td>
      <td>China</td>
      <td>Historical</td>
      <td>Asia</td>
      <td>Asian Literature</td>
      <td>Adult</td>
      <td>Adult Fiction</td>
      <td>Audiobook</td>
      <td>Novels</td>
      <td>0.36</td>
      <td>0.33</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>307364.0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>Hardcover</td>
      <td>38.0</td>
      <td>6 - 9 years</td>
      <td>1060L</td>
      <td>1 - 4</td>
      <td>7121.39</td>
      <td>3.90</td>
      <td>4.6</td>
      <td>4330</td>
      <td>103</td>
      <td>20.32</td>
      <td>1.17</td>
      <td>29.21</td>
      <td>Picture Books</td>
      <td>Biography</td>
      <td>Nonfiction</td>
      <td>Childrens</td>
      <td>History</td>
      <td>Adventure</td>
      <td>Biography Memoir</td>
      <td>Historical</td>
      <td>Womens</td>
      <td>Juvenile</td>
      <td>0.35</td>
      <td>0.21</td>
      <td>0.14</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>274159.0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>Hardcover</td>
      <td>544.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1564.89</td>
      <td>4.31</td>
      <td>4.5</td>
      <td>799</td>
      <td>36</td>
      <td>17.78</td>
      <td>4.80</td>
      <td>24.71</td>
      <td>Cookbooks</td>
      <td>Cooking</td>
      <td>Nonfiction</td>
      <td>Food</td>
      <td>Reference</td>
      <td>Food Writing</td>
      <td>Essays</td>
      <td>Memoir</td>
      <td>Foodie</td>
      <td>Culinary</td>
      <td>0.39</td>
      <td>0.19</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>489992.0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Hardcover</td>
      <td>120.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5307.03</td>
      <td>4.25</td>
      <td>3.5</td>
      <td>8192</td>
      <td>28</td>
      <td>15.24</td>
      <td>1.27</td>
      <td>22.86</td>
      <td>Graphic Novels</td>
      <td>Fantasy</td>
      <td>Angels</td>
      <td>Young Adult</td>
      <td>Manga</td>
      <td>Paranormal</td>
      <td>Romance</td>
      <td>Paranormal Romance</td>
      <td>Supernatural</td>
      <td>Comics</td>
      <td>0.30</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>432966.0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 34 columns</p>
</div>




```python
test = datauji
```


```python
ord_enc = OrdinalEncoder()
test[['lexile_measure']] = ord_enc.fit_transform(test[["lexile_measure"]])
test[['reading_age']] = ord_enc.fit_transform(test[["reading_age"]])
test[['grade_level']] = ord_enc.fit_transform(test[["grade_level"]])
```


```python
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
test[['lexile_measure']] = labelencoder.fit_transform(test[["lexile_measure"]])
test[['reading_age']] = labelencoder.fit_transform(test[["reading_age"]])
test[['grade_level']] = labelencoder.fit_transform(test[["grade_level"]])
test['bookformat'] = labelencoder.fit_transform(test['bookformat'])
test['genre_0'] = labelencoder.fit_transform(test['genre_0'])
test['genre_1'] = labelencoder.fit_transform(test['genre_1'])
test['genre_2'] = labelencoder.fit_transform(test['genre_2'])
test['genre_3'] = labelencoder.fit_transform(test['genre_3'])
test['genre_4'] = labelencoder.fit_transform(test['genre_4'])
test['genre_5'] = labelencoder.fit_transform(test['genre_5'])
test['genre_6'] = labelencoder.fit_transform(test['genre_6'])
test['genre_7'] = labelencoder.fit_transform(test['genre_7'])
test['genre_8'] = labelencoder.fit_transform(test['genre_8'])
test['genre_9'] = labelencoder.fit_transform(test['genre_9'])
test
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\sklearn\utils\validation.py:63: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(*args, **kwargs)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookformat</th>
      <th>pages</th>
      <th>reading_age</th>
      <th>lexile_measure</th>
      <th>grade_level</th>
      <th>weight</th>
      <th>rating_value_0</th>
      <th>rating_value_1</th>
      <th>rating_count_0</th>
      <th>rating_count_1</th>
      <th>...</th>
      <th>genre_1_weight</th>
      <th>genre_2_weight</th>
      <th>genre_3_weight</th>
      <th>genre_4_weight</th>
      <th>genre_5_weight</th>
      <th>genre_6_weight</th>
      <th>genre_7_weight</th>
      <th>genre_8_weight</th>
      <th>genre_9_weight</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4</td>
      <td>295.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>3628.74</td>
      <td>3.79</td>
      <td>4.4</td>
      <td>3412</td>
      <td>178</td>
      <td>...</td>
      <td>0.20</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>129789.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>353.0</td>
      <td>20</td>
      <td>33</td>
      <td>11</td>
      <td>521.63</td>
      <td>3.88</td>
      <td>4.1</td>
      <td>168718</td>
      <td>2908</td>
      <td>...</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>262465.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>373.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>5443.10</td>
      <td>3.96</td>
      <td>4.5</td>
      <td>6845</td>
      <td>304</td>
      <td>...</td>
      <td>0.16</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>182195.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>423.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>680.39</td>
      <td>4.07</td>
      <td>4.6</td>
      <td>30037</td>
      <td>1887</td>
      <td>...</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>288596.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>310.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>453.59</td>
      <td>4.23</td>
      <td>4.4</td>
      <td>9193</td>
      <td>463</td>
      <td>...</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>230270.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>495</td>
      <td>1</td>
      <td>448.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>771.11</td>
      <td>4.15</td>
      <td>4.6</td>
      <td>7953</td>
      <td>409</td>
      <td>...</td>
      <td>0.22</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>352263.0</td>
    </tr>
    <tr>
      <td>496</td>
      <td>1</td>
      <td>589.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>748.43</td>
      <td>3.65</td>
      <td>3.8</td>
      <td>39510</td>
      <td>2920</td>
      <td>...</td>
      <td>0.33</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>307364.0</td>
    </tr>
    <tr>
      <td>497</td>
      <td>1</td>
      <td>38.0</td>
      <td>29</td>
      <td>2</td>
      <td>1</td>
      <td>7121.39</td>
      <td>3.90</td>
      <td>4.6</td>
      <td>4330</td>
      <td>103</td>
      <td>...</td>
      <td>0.21</td>
      <td>0.14</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>274159.0</td>
    </tr>
    <tr>
      <td>498</td>
      <td>1</td>
      <td>544.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>1564.89</td>
      <td>4.31</td>
      <td>4.5</td>
      <td>799</td>
      <td>36</td>
      <td>...</td>
      <td>0.19</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>489992.0</td>
    </tr>
    <tr>
      <td>499</td>
      <td>1</td>
      <td>120.0</td>
      <td>20</td>
      <td>15</td>
      <td>11</td>
      <td>5307.03</td>
      <td>4.25</td>
      <td>3.5</td>
      <td>8192</td>
      <td>28</td>
      <td>...</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.11</td>
      <td>0.10</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>432966.0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 34 columns</p>
</div>



# Prediksi Data Uji


```python
uji_X = test.drop(labels='price',axis=1)
uji_Y = test['price']
```


```python
preduji=rfc1.predict(uji_X)
```


```python
RMSEuji = np.sqrt(np.mean(pow(preduji - uji_Y, 2)))
RMSEuji
```




    204803.69925020885



# Final Visualisation


```python
sns.countplot(preduji)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12c7f864fc8>




![png](output_90_1.png)



```python
sns.distplot(preduji)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12d62b7aa08>




![png](output_91_1.png)


# Export to CSV


```python
preduji1[['price']] = pd.DataFrame(preduji)
```


```python
preduji1[['price']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>173100.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>274159.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>166603.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>212224.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>144226.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
preduji1[['price']].to_csv('pred uji final.csv', sep = ',')
```


```python
hasil=pd.read_csv('pred uji_edit.csv', sep=';')
```


```python
hasil.to_csv('pred uji fix.csv', sep=',', index=False)
```
