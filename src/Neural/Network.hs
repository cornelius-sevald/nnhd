module Neural.Network
  ( Network(..)
  , newNetwork
  , toLayers
  )
where

import           System.Random
import           Foreign.Storable               ( Storable )
import           Control.Monad.State
import qualified Data.Semigroup                as Semigroup
import qualified Data.Vector                   as V
import           Numeric.LinearAlgebra

import           Neural.Layer                   ( Layer(..)
                                                , newLayer
                                                )

-- | Simple feed-forward network.
data Network a = Network
    { inputLayer   :: Layer a
    , hiddenLayers :: V.Vector (Layer a)
    , outputLayer  :: Layer a
    } deriving (Show)

instance Semigroup.Semigroup (Network a) where
  (Network il1 hls1 ol1) <> (Network il2 hls2 ol2) =
    let il  = il1
        hls = hls1 `V.snoc` ol1 `V.snoc` il2 V.++ hls2
        ol  = ol2
    in  Network ol hls ol

(<->) :: (Semigroup.Semigroup a) => a -> a -> a
(<->) = (Semigroup.<>)

newNetwork
  :: (RandomGen g, Random a, Element a, Floating a)
  => [Int]
  -> State g (Network a)
newNetwork []    = undefined
newNetwork sizes = do
  let sizes' = zip sizes (tail sizes ++ [0])
  layers <- mapM (uncurry newLayer) sizes'
  return $ fromLayers layers

-- | Create a network from a list of layers.
fromLayers :: [Layer a] -> Network a
fromLayers []  = error "Network needs at least 2 layers, 0 were given."
fromLayers [x] = error "Network needs at least 2 layers, 1 were given."
fromLayers (x : xs) =
  let il  = x
      hls = init xs
      ol  = last xs
  in  Network il (V.fromList hls) ol

-- | Convert the network to a list of layers.
toLayers :: Network a -> [Layer a]
toLayers (Network il hls ol) = V.toList $ il `V.cons` hls `V.snoc` ol
