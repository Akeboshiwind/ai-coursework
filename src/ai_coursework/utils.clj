(ns ai-coursework.utils)

(let [id-source (atom 0)]
  (defn new-id
    []
    (swap! id-source inc)))

(defn fmap
  [f m]
  (zipmap (keys m)
          (map f (vals m))))

(defn invert-map
  [map]
  (into {}
        (reduce (fn [m [k v]]
                  (merge-with concat
                              m
                              (reduce (fn [m v] (assoc m v [k]))
                                      {}
                                      v)))
                {}
                map)))

(defn sigmoid
  [x]
  (/ 1 (+ 1 (Math/exp (- x)))))

(defn sigmoid-deriv
  [x]
  (* x (- 1 x)))
