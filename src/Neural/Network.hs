{-# LANGUAGE FlexibleContexts #-}
module Neural.Network
  ( Network(..)
  , newNetwork
  , sendforwards
  , sendforward
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

-- | Propagate an input through the network
-- and return every intermediate weighted input along the way including
-- the original input.
sendforwards
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> [Vector a]            -- ^ The weighted inputs of all the layers
sendforwards activation input (Network layers) =
    let f  = feed activation
        g  = send
        az = scanl (\(x, x') l -> (f x l, g x l)) (input, input) layers
     in map snd az

-- | Propagate an input through the network
-- and return the weighted input of the last layer.
sendforward
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> Vector a              -- ^ The wighted input of the last layer
sendforward activation input = last . sendforwards activation input

-- | Propagate an input through the network
-- and return every intermediate activation along the way including
-- the original input.
feedforwards
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> [Vector a]            -- ^ The activations of all the layers
feedforwards activation input (Network layers) =
    scanl (feed activation) input layers

-- | Propagate an input through the network
-- and return the activation of the last layer.
feedforward
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> Vector a              -- ^ The activations of the last layer
feedforward activation input = last . feedforwards activation input
