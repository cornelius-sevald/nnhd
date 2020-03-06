{-# LANGUAGE FlexibleContexts #-}
module Neural.Network
  ( Network(..)
  , newNetwork
  , feedforwards
  , feedforward
  )
where

import           System.Random
import           Foreign.Storable               ( Storable )
import           Control.Monad.State
import qualified Data.Semigroup                as Semigroup
import           Numeric.LinearAlgebra

import           Neural.Activation              ( ActivationFunction(..))
import           Neural.Layer

-- | Simple feed-forward network.
newtype Network a = Network [Layer a]
    deriving (Show)

-- | Construct a new random network given the sizes of the layers
-- Each bias and weight is given a random value normally
-- distributed with mean 0 and standard deviation 1.
newNetwork
  :: (RandomGen g, Random a, Element a, Floating a)
  => [Int]    -- ^ The sizes of the layers
  -> State g (Network a)
newNetwork sizes = do
  let sizes' = zip (init sizes) (tail sizes)
  layers <- mapM (uncurry newLayer) sizes'
  return $ Network layers

-- | Feed forward an input through the network
-- and return every intermediate along the way.
feedforwards
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> [Vector a]            -- ^ All of the values through the network
feedforwards activation input (Network layers) =
    scanl (forward activation) input layers

-- | Feed forward an input through the network
-- and return the output of the last layer.
feedforward
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> Vector a              -- ^ The network output
feedforward activation input = last . feedforwards activation input
