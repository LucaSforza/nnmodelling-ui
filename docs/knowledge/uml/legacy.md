# Visual Paradigm reference transcription

## Purpose

Record what original `analysis/uml/nn.vpp` and generated
`analysis/uml/nn_metamodel.puml` actually contain. This is historical reference,
not the native class design. The `.vpp` is SQLite; its 25 Class records include
unnamed placeholders and duplicates. Generated PlantUML keeps 15 named
concepts, nine named associations, and nine Node generalizations. It has no
operation records. Spelling `ParameterIstance` and the unusual handle-as-Node
generalizations are retained here as provenance, then reconciled in
[current metamodel](metamodel.md). The VPP also contains one `Agent` actor,
15 named use cases, one extend relation, and `Training`/`Modellazione` groups.
Its 17 Constraint records and 12 NOTE records have no usable names; the
legacy [MCP use-case parity document](https://github.com/LucaSforza/NNModelling/blob/master/docs/knowledge/uml/mcp-use-case-parity.md)
provides their accepted interpretation.

## Diagram

```mermaid
classDiagram
    class Node {
      +string id
      +string name
      +string type
    }
    class Input
    class Layer
    class Join
    class Module
    class Subflow
    class LayerConnection
    class ExternalConnection
    class SourceHandle
    class TargetHandle
    class Connection
    class Handle
    class Stereotype
    class Parameters
    class ParameterIstance
    class StereotypeApplication
    Node <|-- Input
    Node <|-- Layer
    Node <|-- Join
    Node <|-- Module
    Node <|-- Subflow
    Node <|-- LayerConnection
    Node <|-- ExternalConnection
    Node <|-- SourceHandle
    Node <|-- TargetHandle
    Subflow "1" --> "0..*" Node : mod_lay
    Input "0..1" --> "0..*" Layer : ini_lay
    Handle "0..*" --> Node : nod_han
    Connection "0..*" --> "1" SourceHandle : source_handle
    Connection "0..1" --> "1" TargetHandle : target_handle
    Stereotype "1" --> "0..*" Parameters : par_ster
    StereotypeApplication "1" --> "0..*" ParameterIstance : ster_par_istance
    ParameterIstance "0..*" --> "1" Parameters : istance_par
    Stereotype "1" --> "0..*" Module : ster_mod
```

## Legacy use cases

This second VPP model is transcribed in original terms. Associations and the
layout-before-screenshot rule follow the accepted legacy parity document;
grouping alone does not imply the native client implements training.

```mermaid
flowchart LR
    A[Agent]
    subgraph Modellazione
      M1([Aggiungere nodo canvas])
      M2([Collegare nodi])
      M3([Modificare parametri nodi])
      M4([Formattare la vista])
      M5([Screenshot])
      M6([Aprire un progetto])
      M7([Creare Steriotipo])
      M8([Eliminare Steriotipo])
      M9([Creare dataset])
      M10([Eliminare Dataset])
    end
    subgraph Training
      T1([Collegarsi al backend])
      T2([Modificare parametri])
      T3([Lanciare Training])
      T4([Monitoraggio training])
      T5([Download wheel])
    end
    A --- M1
    A --- M2
    A --- M3
    A --- M4
    A --- M5
    A --- M6
    A --- M7
    A --- M8
    A --- M9
    A --- M10
    A --- T1
    A --- T2
    A --- T3
    A --- T4
    A --- T5
    M4 -. "extend: arrange before capture" .-> M5
```

## Reconciliation and constraints

The generated file additionally generalizes `Node` to both handle types and
both connection-like types, although it also models `Connection` and `Handle`
separately. Those inconsistent references are not treated as a current
inheritance contract. The unlabelled multiplicity at `Handle→Node` is
unspecified. No operations or semantic constraints can be recovered faithfully
from this VPP model; current operations and constraints come from accepted
legacy graph/package contracts and are marked as such in normative UML.

Formal reference constraint: `∀a∈transcribedAssociations:
a.name,a.ends,a.multiplicity = generatedPuml(a)` where present. Natural: this
diagram preserves named source relationships without inventing missing ones.

No C mapping applies to historical inheritance. Current tagged/composed C
mapping appears in [metamodel](metamodel.md).
