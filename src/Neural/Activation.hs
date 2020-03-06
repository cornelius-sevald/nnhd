module Neural.Activation
  ( ActivationFunction(..)
  , ActivationFunction'(..)
  , sigmoid
  , sigmoid'
  , reLU
  , reLU'
  )
where

type ActivationFunction a = a -> a
type ActivationFunction' a = a -> a

-- | The sigmoid activation function.
-- Useful when computing probabilities,
-- as the result is always between 0 and 1.
sigmoid :: Floating a => ActivationFunction a
sigmoid x = 1 / (x + exp (negate x))

-- | The derivative of the `sigmoid` function.
sigmoid' :: Floating a => ActivationFunction' a
sigmoid' x = y - y ^ 2 where y = sigmoid x

-- | The Rectified Linear Unit function.
-- This is a very popular activation function, as it is very fast.
reLU :: (Num a, Ord a) => ActivationFunction a
reLU = max 0

-- | The derivative of the Rectified Linear Unit function `reLU`.
-- The derivative at 0 has been arbitrarily chosen to equal 1 instead of 0.
reLU' :: (Num a, Ord a) => ActivationFunction' a
reLU' x | x < 0     = 0
        | otherwise = 1
