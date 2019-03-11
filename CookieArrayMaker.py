#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:52:18 2019

@author: mackenzieobrien
"""
import numpy as np
def main():
#    ingredientNames=['Sugar (c)', 'baking powder (tsp)', 'Flour (c)', 
#        'chocolate chunks (oz)', 'egg yolk', 'Egg White', 'egg substitute (c)', 
#        'Egg', 'melted butter (c)', 'Butter (c)', 'Brown Sugar (c)', 
#        'Vanilla (tsp)', 'BakingSoda(tsp)', 'salt (tsp)', 'coarse salt (tsp)', 
#        'shortening (c)', 'semisweet chocolate chips (oz)', 
#        'melted semisweet chocolate chips (oz)', 'oats (c)', 'macadamia nuts chopped (c)', 
#        'walnuts (c)', 'melted dark chocolate (oz)', 'dark chocolate chips (oz)', 
#        'chocolate covered caramel candy (pieces)', 'chocolate covered toffee (oz)', 
#        'PB chips (c)', 'white chocolate chips (oz)', 'creamy PB (oz)', 'water (tsp)', 
#        'hot water (tsp)', 'chopped pecans (c)', 'cocoa powder (c)', 
#        'milk chocolate chips (c)', 'ground ginger (tsp)', 'milk (tbsp)', 'crips rice cereal (c)', 
#        'chocolate syrup (c)', 'ground cinnamon (tsp)', 'basic cookie mix', 'ground cloves (tsp)', 
#        'ground nutmeg (tsp)', 'vanilla chips (oz)', 'bread flour (c)', 
#        'coconut extract (tsp)', 'flaked coconut (c)', 'applesauce (c)', 'peppermint extract (tsp)', 
#        'crushed pistachios (c)', 'pumpkin puree (oz)', 'mini marshmallows (c)', 
#        'raisins (c)', 'corn flakes (c)', 'coconut oil (tbsp)', 'yellow cake mix (pkg)', 
#        'white cake mix (pkg)', 'sour cream (c)', 'vegetable oil (tbsp)', 
#        'instant butterscotch pudding mix', 'confectioners sugar (c)', 
#        'powdered milk (c)', 'dried tart cranberries (c)', 'almond extract (tsp)', 
#        'honey (tbsp)', 'cream cheese (oz)', '3 ounce package instant pistachio pudding mix', 
#        'mint chocolate chips (c)','Kentucky Bourbon (tbsp)', 'spice cake mix (oz)', 
#        'corn syrup (tbsp)','green tea powder [matcha] (tbsp)', 'bitter sweet chocolate (c)', 
#        'caramel morsels (pieces)', 'molasses (c)', '(3.5 oz)vanilla pudding mix', 
#        'cream of tartar (tsp)', 'white chocolate pudding mix (3.3 oz pkg)', 
#        'plain yogurt', 'carrots shredded (c)', 'dried cranberries (c)', 
#        'sifted whole wheat pastry flour (c)', 'chopped almonds (c)', 'cake flour (c)', 
#        'red food coloring (tbsp)', 'sweet potato puree (c)', 'orange juice (tbsp)', 
#        'gluten free flour (c)', 'almond flour (c)', 'Stevia Extract In The Raw (c)', 
#        'cornstarch (tbsp)', 'kiwi pureed', 'toasted oat cereal smashed (c)', 'cream (tbsp)', 
#        'Chocolate Chip Muffin Mix (pkg)', 'unsweetened baking chocolate (oz)', 
#        'Sweetened Condensed Milk (oz)', 'maple-cured bacon chopped (piece)', 
#        'coconut sugar (c)', 'chia seeds (c)', 'ground flax seeds (c)', 
#        'cocoa nibs (c)', 'chocolate fudge cake mix (pkg)', 'soybean oil (tbsp)', 
#        'butterscotch-flavored chips (c)', 'chopped maraschino cherries (c)', 
#        'coffee flavored liqueur (tbsp)', 'powdered protein supplement (scoop)', 
#        'port wine (c)', 'sorghum flour (c)', 'white rice flour (c)', 'xanthan gum (tsp)', 
#        'hemp seed hearts (c)', 'coconut flour (c)', 'maple syrup (c)', 
#        'ground grahamrackerrumbs (c)', '(18.25 ounce) package chocolate chip cake mix with pudding', 
#        'pumpkin pie spice (tsp)', 'almond butter (c)', 'maple extract (tsp)', 
#        'finely chopped zucchini (c)','half and half (tbsp)', 
#        'dairy-free and gluten-free chocolate chips', 'nut and seed trail mix (c)', 
#        '(1 ounce) squares German sweet chocolate - chopped', 'matzoake meal (c)', 
#        'firmly packed potato starch (c)', 'crunchy peanut butter (c)', 
#        'mashed bananas (c)', 'European cookie spread (c)', 
#        'instant espresso coffee powder (tbsp)', 'cornflakes cereal - curmbled (c)', 
#        'ground mace (tsp)', 'mashed avocado (c)', 'finely chopped crystallized ginger (c)', 
#        'NESTLE TOLL HOUSE Delightfulls Mint Filled Morsels (c)', 
#        'toffee baking bits (c)', 'Tennessee whiskey (tbsp)', 'tapioca flour (c)', 
#        'pureed prunes (tbsp)', 'orange extract (tsp)', 'gluten-free baking mix (c)', 
#        'buttermilk (c)', 'ground allspice (tsp)', 'orange zest (tbsp)', 
#        'ground white pepper (tsp)', 'chickpea flour (c)', 'vegan choc. Chips (tbsp)', 
#        'canola oil (tbsp)', 'soy milk (c)', 'milk chocolate candy kisses - unwrapped - softened', 
#        'blueberries (c)', 'finely chopped chipotle peppers in adobo sauce (tsp)', 
#        'ground cardamom (tsp)', 'almond milk (tbsp)', 'turbinado sugar (c)', 
#        'chopped dried apricot (c)', '(6 ounce) package almond brickle chips', 
#        'candy-coated chocolate pieces (c)', 'white vinegar (tsp)', 
#        'chopped peppermint candy (c)', 'prepared granola (c)', 
#        'buttermilk baking mix (c)', 'golden raisins (c)']
    resultArray=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  #  print len(resultArray)
    ingRange=[2.5,4,12,16,2,4,0.25,4,1,2,3,6,4,2,1,3,40,0.324,5,1,3,8,24,48,8.4,2,16,16,36,12,1.5,1,3,1.5,64,4.5,.33,3,2,1,
         1,12,2,1,2,1,3,1.5,16,1.5,1.5,1,16,1,1,0.66,16,1,3,0.25,1,1.5,8,8,1,1.66,5,18.25,8,1,1,24,0.5,2,4,1,0.5,0.25,1.25,3,
         1.5,0.5,2,0.25,0.66,2,3,0.75,8.11,1,1,4,2,5,14,6,1,0.25,0.25,.5,1,1,0.25,0.5,3,4,
         0.25,1.25,1.25,3,1,1,0.5,1,1,1.5,2,1,1.5,2,0.5,0.75,12,1,0.5,1,1,0.5,5.28,1,0.5,0.5,
         0.75,1.5,0.75,1,0.33,2,1,2.25,0.5,0.5,1,1,2,8,2,0.33,48,0.5,2,1,1,0.5,1,1,0.33,0.5,0.5,2.5,2,1]
#print(len(ingRange))
#print(len(ingredientNames))
    #print(round(np.random.uniform(0,ingRange[0]),2))
    
    CookiesArray = [0] * 100
    for i in range(len(CookiesArray)):
        i=0
        for i in range(len(ingRange)):
            randNum=round(np.random.uniform(0,ingRange[i]),2)
            rounded=round_of_rating(randNum)
            #print(rounded)
            resultArray[i]=rounded
    print(CookiesArray)

def round_of_rating(number):
    """Round a number to the closest half integer.
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0"""
    rounded=round(number*4)
    return rounded/ 4
