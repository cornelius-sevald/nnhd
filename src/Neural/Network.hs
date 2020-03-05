module Neural.Network
  ( Network(..)
  , newNetwork
  )
where

import           System.Random
import           Foreign.Storable               ( Storable )
import           Control.Monad.State
import qualified Data.Semigroup                as Semigroup
import           Numeric.LinearAlgebra

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
  let sizes' = zip sizes (tail sizes ++ [0])
  layers <- mapM (uncurry newLayer) sizes'
  return $ Network layers
