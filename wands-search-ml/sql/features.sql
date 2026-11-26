-- DuckDB SQL query to join WANDS labels, queries, and products,
-- map text relevance labels to numeric scores, and compute catalog & text stats.
-- Expects views/tables: `labels`, `queries`, `products`.

WITH category_counts AS (
    SELECT
        COALESCE(product_class, 'Unknown') AS product_class_clean,
        COUNT(*) AS category_product_count
    FROM products
    GROUP BY COALESCE(product_class, 'Unknown')
),
product_stats AS (
    SELECT
        p.product_id,
        COALESCE(p.product_name, '') AS product_name,
        COALESCE(p.product_description, '') AS product_description,
        COALESCE(p.product_class, 'Unknown') AS product_class,
        CASE
            WHEN TRIM(COALESCE(p.product_name, '')) = '' THEN 0
            ELSE LEN(REGEXP_SPLIT_TO_ARRAY(TRIM(p.product_name), '\s+'))
        END AS title_word_count,
        CASE
            WHEN TRIM(COALESCE(p.product_description, '')) = '' THEN 0
            ELSE LEN(REGEXP_SPLIT_TO_ARRAY(TRIM(p.product_description), '\s+'))
        END AS desc_word_count,
        cc.category_product_count
    FROM products p
    LEFT JOIN category_counts cc
        ON COALESCE(p.product_class, 'Unknown') = cc.product_class_clean
),
query_stats AS (
    SELECT
        q.query_id,
        COALESCE(q.query, '') AS query,
        COALESCE(q.query_class, 'Unknown') AS query_class,
        CASE
            WHEN TRIM(COALESCE(q.query, '')) = '' THEN 0
            ELSE LEN(REGEXP_SPLIT_TO_ARRAY(TRIM(q.query), '\s+'))
        END AS query_word_count
    FROM queries q
)
SELECT
    l.id,
    l.query_id,
    l.product_id,
    l.label AS label_text,
    CASE l.label
        WHEN 'Exact' THEN 2
        WHEN 'Partial' THEN 1
        WHEN 'Irrelevant' THEN 0
        ELSE NULL
    END AS relevance,
    qs.query,
    qs.query_class,
    qs.query_word_count,
    ps.product_name,
    ps.product_description,
    ps.product_class,
    ps.title_word_count,
    ps.desc_word_count,
    ps.category_product_count
FROM labels l
INNER JOIN query_stats qs
    ON l.query_id = qs.query_id
INNER JOIN product_stats ps
    ON l.product_id = ps.product_id
WHERE l.label IN ('Exact', 'Partial', 'Irrelevant');

