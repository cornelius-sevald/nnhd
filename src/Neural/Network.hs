{-# LANGUAGE FlexibleContexts #-}
module Neural.Network
  ( Network(..)
  , newNetwork
  , feedforward
  )
where

import           System.Random
import           Foreign.Storable               ( Storable )
import           Control.Monad.State
import qualified Data.Semigroup                as Semigroup
import           Numeric.LinearAlgebra

import           Neural.Activation              ( ActivationFunction(..))
import           Neural.Layer                   ( Layer(..)
                                                , newLayer
                                                )

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

-- | Feed forward an input through the network.
feedforward
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a  -- ^ The activation function
  -> Vector a              -- ^ The network input
  -> Network a             -- ^ The network
  -> Vector a              -- ^ The network output
feedforward activation input (Network layers) =
  let _weights = map weights layers
      _biases  = map biases layers
      wb       = zip _weights _biases
      f        = cmap activation
      forward x (w, b) = f (w #> x + b)
  in  foldl forward input wb
