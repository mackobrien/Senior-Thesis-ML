import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

df = pd.read_csv('ingredients.csv')

print(df.head())

print(df.describe())

inCol = ['Sugar (c)', 'baking powder (tsp)', 'Flour (c)', 
        'chocolate chunks (oz)', 'egg yolk', 'Egg White', 'egg substitute (c)', 
        'Egg', 'melted butter (c)', 'Butter (c)', 'Brown Sugar (c)', 
        'Vanilla (tsp)', 'BakingSoda(tsp)', 'salt (tsp)', 'coarse salt (tsp)', 
        'shortening (c)', 'semisweet chocolate chips (oz)', 
        'melted semisweet chocolate chips (oz)', 'oats (c)', 'macadamia nuts chopped (c)', 
        'walnuts (c)', 'melted dark chocolate (oz)', 'dark chocolate chips (oz)', 
        'chocolate covered caramel candy (pieces)', 'chocolate covered toffee (oz)', 
        'PB chips (c)', 'white chocolate chips (oz)', 'creamy PB (oz)', 'water (tsp)', 
        'hot water (tsp)', 'chopped pecans (c)', 'cocoa powder (c)', 
        'milk chocolate chips (c)', 'ground ginger (tsp)', 'milk (tbsp)', 'crips rice cereal (c)', 
        'chocolate syrup (c)', 'ground cinnamon (tsp)', 'basic cookie mix', 'ground cloves (tsp)', 
        'ground nutmeg (tsp)', 'vanilla chips (oz)', 'bread flour (c)', 
        'coconut extract (tsp)', 'flaked coconut (c)', 'applesauce (c)', 'peppermint extract (tsp)', 
        'crushed pistachios (c)', 'pumpkin puree (oz)', 'mini marshmallows (c)', 
        'raisins (c)', 'corn flakes (c)', 'coconut oil (tbsp)', 'yellow cake mix (pkg)', 
        'white cake mix (pkg)', 'sour cream (c)', 'vegetable oil (tbsp)', 
        'instant butterscotch pudding mix', 'confectioners sugar (c)', 
        'powdered milk (c)', 'dried tart cranberries (c)', 'almond extract (tsp)', 
        'honey (tbsp)', 'cream cheese (oz)', '3 ounce package instant pistachio pudding mix', 
        'creme de menthe (tbsp)', 'mint chocolate chips (c)', 'Hachiya persimmons pulp extracted', 
        'Kentucky Bourbon (tbsp)', 'spice cake mix (oz)', 'corn syrup (tbsp)', 
        'green tea powder [matcha] (tbsp)', 'bitter sweet chocolate (c)', 
        'caramel morsels (pieces)', 'molasses (c)', '(3.5 oz)vanilla pudding mix', 
        'cream of tartar (tsp)', 'white chocolate pudding mix (3.3 oz pkg)', 
        'plain yogurt', 'carrots shredded (c)', 'driedranberries (c)', 
        'sifted whole wheat pastry flour (c)', 'chopped almonds (c)', 'cake flour (c)', 
        'red food coloring (tbsp)', 'sweet potato puree (c)', 'orange juice (tbsp)', 
        'gluten free flour (c)', 'almond flour (c)', 'Stevia Extract In The Raw (c)', 
        'cornstarch (tbsp)', 'kiwi pureed', 'toasted oat cereal smashed (c)', 'cream (tbsp)', 
        'Chocolate Chip Muffin Mix (pkg)', 'unsweetened baking chocolate (oz)', 
        'Sweetened Condensed Milk (oz)', 'maple-cured bacon chopped (piece)', 
        'coconut sugar (c)', 'chia seeds (c)', 'ground flax seeds (c)', 
        'cocoa nibs (c)', 'chocolate fudge cake mix (pkg)', 'soybean oil (tbsp)', 
        'butterscotch-flavored chips (c)', 'chopped maraschino cherries (c)', 
        'coffee flavored liqueur (tbsp)', 'powdered protein supplement (scoop)', 
        'port wine (c)', 'sorghum flour (c)', 'white rice flour (c)', 'xanthan gum (tsp)', 
        'hemp seed hearts (c)', 'coconut flour (c)', 'maple syrup (c)', 
        'ground grahamrackerrumbs (c)', '(18.25 ounce) package chocolate chip cake mix with pudding', 
        'pumpkin pie spice (tsp)', 'almond butter (c)', 'maple extract (tsp)', 
        'finely chopped zucchini (c)', '1.5', 'half and half (tbsp)', 
        'dairy-free and gluten-free chocolate chips', 'nut and seed trail mix (c)', 
        '(1 ounce) squares German sweet chocolate - chopped', 'matzoake meal (c)', 
        'firmly packed potato starch (c)', 'crunchy peanut butter (c)', 
        'mashed bananas (c)', 'European cookie spread (c)', 
        'instant espresso coffee powder (tbsp)', 'cornflakes cereal - curmbled (c)', 
        'ground mace (tsp)', 'mashed avocado (c)', 'finely chopped crystallized ginger (c)', 
        'NESTLE TOLL HOUSE Delightfulls Mint Filled Morsels (c)', 
        'toffee baking bits (c)', 'Tennessee whiskey (tbsp)', 'tapioca flour (c)', 
        'pureed prunes (tbsp)', 'orange extract (tsp)', 'gluten-free baking mix (c)', 
        'buttermilk (c)', 'ground allspice (tsp)', 'orange zest (tbsp)', 
        'ground white pepper (tsp)', 'chickpea flour (c)', 'vegan choc. Chips (tbsp)', 
        'canola oil (tbsp)', 'soy milk (c)', 'milk chocolate candy kisses - unwrapped - softened', 
        'blueberries (c)', 'finely chopped chipotle peppers in adobo sauce (tsp)', 
        'ground cardamom (tsp)', 'almond milk (tbsp)', 'turbinado sugar (c)', 
        'chopped dried apricot (c)', '(6 ounce) package almond brickle chips', 
        'candy-coated chocolate pieces (c)', 'white vinegar (tsp)', 
        'chopped peppermint candy (c)', 'prepared granola (c)', 
        'buttermilk baking mix (c)', 'golden raisins (c)']




y = df['Rating'].astype(float).as_matrix()
X = df[inCol].astype(float).as_matrix()

## linear regression
reg = GradientBoostingRegressor().fit(X, y)
print(reg.score(X, y))

cc = reg.feature_importances_
for x in range(len(cc)):
  print(str(cc[x]) + " -  " + inCol[x])


print(reg.predict(np.array([[0,0,3,0,0,0,0,2,0,1,0.75,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
print(reg.predict(np.array([[1,0,1.5,0,0,0,0,2,0,1,1.5,1.5,1.25,1,0,0,16,0,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,0,0.125,0.125,0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))



## generating random recipe
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
bestRank = -1000000
bestRecipe = []

f = 'GBReg-res.csv'
if os.path.exists(f):
  rdf = pd.read_csv(f, index_col=0)
else:
  rdf = pd.DataFrame()

while True:
  new = np.abs(np.floor(np.random.normal(mu, sigma)*16)/16)
  rank = reg.predict([new])
  if rank > bestRank:
    bestRank = rank
    bestRecipe = new
    bestDist = 1000000
    for recipe in X:
      dist = np.linalg.norm(recipe-new)
      if dist < bestDist:
        bestDist = dist
    print(bestRecipe)
    br = {}
    br['#Predicted Rank'] = bestRank[0]
    br['#Uniqueness'] = bestDist
    br['#Simplicity'] = np.linalg.norm(new)
    for x in range(len(inCol)):
      br[inCol[x]] = bestRecipe[x]
    #print(br)
    rdf = rdf.append(br, ignore_index=True)
    #print(rdf)
    rdf.to_csv(f)
    print("Rank: " + str(bestRank) + " Distance: " + str(bestDist) + " Norm: " + str(br['#Simplicity']))




