# Tensorflow Tutorials in Hylang

#### Emacs `M-x inferior-lisp` repl eval the example.hy

```clojure
(import tensorflow)
(import numpy)

(setv x_data (.astype (numpy.random.rand 100) numpy.float32))
(setv y_data (+ (* x_data 0.1) 0.3))
(setv Weights (tensorflow.Variable (tensorflow.random_uniform [1] -1.0 1.0)))
(setv biases (tensorflow.Variable (tensorflow.zeros [1])))
(setv y (+ (* x_data Weights) biases))
(setv loss (tensorflow.reduce_mean (tensorflow.square (- y y_data))))
(setv optimizer (tensorflow.train.GradientDescentOptimizer 0.5))
(setv train (optimizer.minimize loss))
(setv sess (tensorflow.Session))
(setv init (tensorflow.global_variables_initializer))
(sess.run init)

(for [step (range 201)]
  (do
   (sess.run train)
   (if (= (% step 20) 0)
     (print step (sess.run Weights) (sess.run biases)))))
;; =>
;; 0 [0.15042791] [0.35226622]
;; 20 [0.10480256] [0.29769197]
;; 40 [0.10112178] [0.2994609]
;; 60 [0.10026202] [0.2998741]
;; 80 [0.10006122] [0.2999706]
;; 100 [0.10001431] [0.29999313]
;; 120 [0.10000335] [0.2999984]
;; 140 [0.1000008] [0.29999962]
;; 160 [0.10000018] [0.29999992]
;; 180 [0.10000011] [0.29999995]
;; 200 [0.10000011] [0.29999995]

(first
 (map (fn [x]
        (do
         (setv y 100)
         (+ x y))) [1 2 5 6])) ;;=> 101L

```

#### 将S表达式编译为python:`hy2py example.hy`,`first, map, range`只是hy.core.language的函数,lambda多行不支持,所以只能单独写一个函数出来

```python

from hy.core.language import first, map, range
import tensorflow
import numpy
x_data = numpy.random.rand(100L).astype(numpy.float32)
y_data = x_data * 0.1 + 0.3
Weights = tensorflow.Variable(tensorflow.random_uniform([1L], -1.0, 1.0))
biases = tensorflow.Variable(tensorflow.zeros([1L]))
y = x_data * Weights + biases
loss = tensorflow.reduce_mean(tensorflow.square(y - y_data))
optimizer = tensorflow.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
sess = tensorflow.Session()
init = tensorflow.global_variables_initializer()
sess.run(init)
for step in range(201L):
    sess.run(train)
    print(step, sess.run(Weights), sess.run(biases)
        ) if step % 20L == 0L else None


def _hy_anon_var_1(x):
    y = 100L
    return x + y


first(map(_hy_anon_var_1, [1L, 2L, 5L, 6L]))

```
