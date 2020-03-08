{-# LANGUAGE FlexibleContexts #-}
module Neural.Network
  ( Network(..)
  , weights
  , biases
  , newNetwork
  , randomNetwork
  , feedforwards
  , feedforward
  )
where

import           System.Random
import           Foreign.Storable               ( Storable )
import           Control.Monad.State
import qualified Data.Semigroup                as Semigroup
import           Numeric.LinearAlgebra

import           Neural.Activation              ( ActivationFunction(..) )
import           Neural.Layer            hiding ( weights
                                                , biases
                                                )
import qualified Neural.Layer                  as Layer

-- | Simple feed-forward network.
newtype Network a = Network [Layer a]
    deriving (Show)

-- | Get the weights of the network.
weights :: Network a -> [Matrix a]
weights (Network layers) = map Layer.weights layers

-- | Get the biases of the network.
biases :: Network a -> [Vector a]
biases (Network layers) = map Layer.biases layers

-- | Construct a new network from a list of weigth matricws and bias vectors.
newNetwork :: [Matrix a] -> [Vector a] -> Network a
newNetwork ws bs = Network $ zipWith newLayer ws bs

-- | Construct a new random network given the sizes of the layers
-- Each bias and weight is given a random value normally
-- distributed with mean 0 and standard deviation 1.
randomNetwork
  :: (RandomGen g, Random a, Element a, Floating a)
  => [Int]    -- ^ The sizes of the layers
  -> State g (Network a)
randomNetwork sizes = do
  let sizes' = zip (init sizes) (tail sizes)
  layers <- mapM (uncurry randomLayer) sizes'
  return $ Network layers

-- | Propagate an input through the network and return every intermediate
-- activation and weighted input along the way.
feedforwards
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a   -- ^ The activation function
  -> Vector a               -- ^ The network input
  -> Network a              -- ^ The network
  -> [(Vector a, Vector a)] -- ^ The activations of all the layers
feedforwards σ x (Network layers) = tail $ scanl (feed σ . fst) (x, x) layers

-- | Propagate an input through the network
-- and return the activation and weighted input of the last layer.
feedforward
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> (Vector a, Vector a)  -- ^ The activations of the last layer
feedforward σ x = last . feedforwards σ x
