{-# LANGUAGE DeriveGeneric, DeriveAnyClass #-}
{-# LANGUAGE FlexibleContexts #-}
module Neural.Layer
  ( Layer(..)
  , newLayer
  , randomLayer
  , feed
  )
where

import           System.Random
import           Data.Random.Normal
import           Control.Monad
import           Control.Monad.State
import           Numeric.LinearAlgebra
import           GHC.Generics                   ( Generic )
import           Control.DeepSeq

import           Neural.Activation              ( ActivationFunction(..) )

-- | A layer in a neural network.
data Layer a = Layer
    { weights :: Matrix a    -- Weight matrix
    , biases  :: Vector a    -- Bias vector
    } deriving (Show, Generic, NFData)

-- | Construct a new layer from a weigth matrix and a bias vector.
newLayer :: Matrix a -> Vector a -> Layer a
newLayer ws bs = Layer { weights = ws, biases = bs }

-- | Construct a new random layer given its,
-- size and the size of the previous layer.
-- Each bias and weight is given a random value
-- normally distributed with mean 0 and standard deviation 1.
randomLayer
  :: (RandomGen g, Random a, Element a, Floating a)
  => Int         -- ^ The size of the previous layer
  -> Int         -- ^ The size of the layer
  -> State g (Layer a)
randomLayer m n = do
  _biases  <- replicateM n $ state normal
  _weights <- replicateM n $ replicateM m $ state normal
  return $ Layer (fromLists _weights) (fromList _biases)

-- | Calculate the activation and weighted input of a layer
feed
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The layer input
  -> Layer a               -- ^ The layer
  -> (Vector a, Vector a)  -- ^ The layer output
feed _σ x (Layer w b) = (σ y, y)
 where
  y = w #> x + b
  σ = cmap _σ
