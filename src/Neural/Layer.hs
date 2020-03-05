module Neural.Layer
  ( Layer(..)
  , newLayer
  )
where

import           System.Random
import           Data.Random.Normal
import           Control.Monad
import           Control.Monad.State
import           Numeric.LinearAlgebra

-- | A layer in a neural network.
data Layer a = Layer
    { weights :: Matrix a    -- Weight matrix
    , biases  :: Vector a    -- Bias vector
    } deriving (Show)

-- | Construct a new random layer given its,
-- size and the size of the next layer.
-- Each bias and weight is given a random value
-- normally distributed with mean 0 and standard deviation 1.
newLayer
  :: (RandomGen g, Random a, Element a, Floating a)
  => Int         -- ^ The size of the layer
  -> Int         -- ^ The size of the next layer
  -> State g (Layer a)
newLayer n m = do
  biases  <- replicateM n $ state normal
  weights <- replicateM n $ replicateM m $ state normal
  return $ Layer (fromLists weights) (fromList biases)
