# Structured Output (Schema-Constrained / Function-Calling Decoding)

Extends [ConstrainedDecoding](../ConstrainedDecoding) from a hardcoded
free-form JSON grammar to a **user-supplied JSON Schema**, the structured-output
/ tool-calling path. `CompileJSONSchemaToGBNF` (in `neural/neuraldecode.pas`)
turns a JSON Schema into a GBNF grammar that the existing `TNNetGrammar`
consumes, and `CreateJSONSchemaConstraint` wraps that grammar in a
`TNNetGrammarConstraint` — the same "allowed next tokens" hook the streamed
generation loop applies to the post-softmax probability row **before** the
sampler.

The schema is a typical tool-call arguments object — think
`get_weather(location, days, unit)`:

```json
{
  "type": "object",
  "properties": {
    "location": { "type": "string" },
    "days":     { "type": "integer" },
    "unit":     { "enum": ["celsius", "fahrenheit"] }
  },
  "required": ["location", "days"],
  "additionalProperties": false
}
```

Because the constraint only ever exposes grammar-legal tokens, even the tiny
**untrained** char-level network here can **only** emit JSON that validates
against the schema: the declared keys in declared order, the right value types
(`location` a string, `days` an integer, `unit` one of the two enum literals),
nothing extra (`additionalProperties:false`), and EOS only once a complete
schema-valid object stands. Every sample is checked back through a fresh
`TNNetGrammar` machine **and** parsed as JSON to prove it.

As in ConstrainedDecoding, a small hand-set bias toward structural characters
(and the closing quote) keeps the sampled values short; it is a stand-in for a
trained model's preferences and plays **no** part in correctness — validity
comes from the compiled grammar alone.

## Supported JSON-Schema subset (v1)

- `object` with `properties` + `required` (ordered property rules; non-required
  props are optional), `additionalProperties: false` (default here) /
  `true`/schema (re-opens the object to extra members)
- `array` with `items` + `minItems` / `maxItems`
- `string` (+ `enum` → literal alternation, + `pattern` → char-class rule)
- `number` / `integer`, `boolean`, `null`
- `anyOf` / `oneOf` → alternation
- `$ref` / `$defs` (and legacy `definitions`) for recursion

**Out of scope for v1** (documented in the `neuraldecode` unit header, not
enforced): `allOf`, `format` validators (date-time / email / uuid / …), and
numeric `minimum` / `maximum` / `multipleOf` bound enforcement (numbers are
matched structurally, not range-checked).

Typical output:

```
=== Schema-constrained decoding (same net, same sampler) ===
  sample 1: {"location":"...","days":3}
     -> schema-valid: yes; parses as JSON: yes
```

Runs in seconds on CPU.
