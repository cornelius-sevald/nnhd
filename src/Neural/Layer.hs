{-# LANGUAGE FlexibleContexts #-}
module Neural.Layer
  ( Layer(..)
  , newLayer
  , forward
  )
where

import           System.Random
import           Data.Random.Normal
import           Control.Monad
import           Control.Monad.State
import           Numeric.LinearAlgebra

import Neural.Activation (ActivationFunction(..))

-- | A layer in a neural network.
data Layer a = Layer
    { weights :: Matrix a    -- Weight matrix
    , biases  :: Vector a    -- Bias vector
    } deriving (Show)

-- | Construct a new random layer given its,
-- size and the size of the previous layer.
-- Each bias and weight is given a random value
-- normally distributed with mean 0 and standard deviation 1.
newLayer
  :: (RandomGen g, Random a, Element a, Floating a)
  => Int         -- ^ The size of the previous layer
  -> Int         -- ^ The size of the layer
  -> State g (Layer a)
newLayer m n = do
  _biases  <- replicateM n $ state normal
  _weights <- replicateM n $ replicateM m $ state normal
  return $ Layer (fromLists _weights) (fromList _biases)

-- | Propagate an input through a layer
forward
    :: (Numeric a, Num (Vector a))
    => ActivationFunction a  -- ^ The activation function
    -> Vector a              -- ^ The layer input
    -> Layer a               -- ^ The layer
    -> Vector a              -- ^ The layer output
forward activation x (Layer w b) = f (w #> x + b)
    where f = cmap activation
