(ns ai-coursework.layers
  (:require [ai-coursework.utils :as u]))

(defn input
  "Create an input layer"
  ([width opts]
   (merge (input width)
          opts))
  ([width]
   {:type :input
    :width width
    :activation-fn {:fn identity :deriv identity}
    :bias-fn (constantly 0)
    :weight-fn (constantly 1)}))

(defn sigmoid
  "Create an sigmoid layer"
  ([width opts]
   (merge (input width)
          opts))
  ([width]
   {:type :sigmoid
    :width width
    :activation-fn {:fn u/sigmoid :deriv u/sigmoid-deriv}
    :bias-fn rand
    :weight-fn rand}))
