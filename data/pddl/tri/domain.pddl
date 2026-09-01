(define (domain tri)
  (:requirements :strips)
  (:predicates (between ?x ?y ?z)
               (link ?x ?y)
               (flag))
  (:action swap
     :parameters (?x ?y ?z)
     :precondition (and (between ?x ?y ?z))
     :effect (and (not (between ?x ?y ?z)) (between ?z ?y ?x))))
