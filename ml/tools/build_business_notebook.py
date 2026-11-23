"""Write notebooks/Driven_Product_and_Business.ipynb: plans, unit economics, pricing, leads and launch readiness.

Every number comes from product.json through the package, so the notebook can never drift from the product.
Rebuild it with ``python tools/check.py --notebooks``.
"""

from __future__ import annotations

from _cells import code, md, save

NOTEBOOK_FILE = "Driven_Product_and_Business.ipynb"

CELLS = [
    md(
        """
        # Driven: product and business

        Plans, unit economics, a price sweep, the weekly leads report and launch readiness. Prices, rates and
        flags all come from `product.json`, the single source the pipeline and the app share.
        """
    ),
    md("## 1. Plans"),
    code(
        """
        from dataclasses import asdict, replace

        import numpy as np
        import pandas as pd

        from driven_ml.product import UsageAssumptions, load_product
        from driven_ml.settings import find_repo_root

        pd.set_option("display.max_colwidth", 140)
        pd.set_option("display.width", 200)
        repo_root = find_repo_root()
        product = load_product(repo_root)
        plans = pd.DataFrame([asdict(plan) for plan in product.plans])
        plans["features"] = plans["features"].map(", ".join)
        plans
        """
    ),
    md(
        """
        ## 2. Unit economics of the Family plan

        Each false alert texts every confirmed contact, so false alerts cost money as well as trust. The
        staging gate checks the margin at the model's measured test false-alert rate; the break-even rate is
        the most false alerts per drive hour the price can absorb at the target margin.
        """
    ),
    code(
        """
        from driven_ml.economics import break_even_false_alert_rate, unit_economics
        from driven_ml.spec import FALSE_ALERT_BUDGET_PER_HOUR

        rates = [0.0, 0.2, 1.0, 2.0, 4.0]
        economics = pd.DataFrame([asdict(unit_economics(product, rate)) for rate in rates])
        display(economics.round(3))
        print(f"Break-even: {break_even_false_alert_rate(product):.2f} false alerts per drive hour "
              f"at a {product.min_contribution_margin:.0%} margin target")
        print("Still team estimates, to replace with measured pilot values:", ", ".join(UsageAssumptions.names()))
        print(product.raw["assumptions"]["note"])
        """
    ),
    md(
        """
        ## 3. Price sweep

        The contribution margin of the Family plan at the 0.2 false alerts per hour budget, for four prices.
        """
    ),
    code(
        """
        def with_family_price(price_usd_month: float):
            plans = tuple(
                replace(plan, price_usd_month=price_usd_month) if plan.plan_id == "family" else plan
                for plan in product.plans
            )
            return replace(product, plans=plans)


        sweep = pd.DataFrame(
            [
                {
                    "price_usd_month": price,
                    "margin_at_budget": unit_economics(
                        with_family_price(price), FALSE_ALERT_BUDGET_PER_HOUR
                    ).contribution_margin,
                    "break_even_false_alerts_per_hour": break_even_false_alert_rate(with_family_price(price)),
                }
                for price in (4.99, 7.99, 9.99, 12.99)
            ]
        )
        sweep.round(3)
        """
    ),
    md(
        """
        ## 4. Leads report

        The landing page's lead export, summarized for the weekly go-to-market review. This example export
        is generated; point `summarize_leads` at the real CSV from the backend instead.
        """
    ),
    code(
        """
        from driven_ml.leads import summarize_leads

        rng = np.random.default_rng(4)
        count = 60
        example = pd.DataFrame(
            {
                "created_at": (
                    pd.Timestamp("2026-10-05", tz="UTC") + pd.to_timedelta(rng.uniform(0, 28 * 24, count), unit="h")
                ).floor("s"),
                "email": [f"lead{i}@example.com" for i in range(count)],
                "role": rng.choice(
                    ["parent", "insurer", "dashcam_maker", "driver_education"], count, p=[0.82, 0.08, 0.05, 0.05]
                ),
                "state": rng.choice(["TX", "OK", "LA", "NM", "CA"], count, p=[0.7, 0.1, 0.08, 0.07, 0.05]),
                "utm_source": rng.choice(
                    ["instagram", "school_flyer", "linkedin", None], count, p=[0.4, 0.3, 0.15, 0.15]
                ),
                "teen_drivers": rng.integers(0, 3, count),
                "sms_marketing_consent": rng.random(count) < 0.45,
            }
        )
        leads = summarize_leads(example)
        print(f"{leads.total_leads} leads, {leads.teen_drivers} teen drivers")
        print(f"{leads.sms_opt_in_rate:.0%} opted in to marketing texts")
        display(leads.by_week)
        display(leads.by_role)
        display(leads.by_source)
        display(leads.by_state)
        leads.partner_leads
        """
    ),
    md(
        """
        ## 5. Launch readiness

        Everything that must be true before real families use Driven. In this ML-only repository most rows
        fail, which is correct: there is no app and no released real-data model yet. Each failing row says
        what it needs.
        """
    ),
    code(
        """
        from driven_ml.launch import readiness_checks

        checks = pd.DataFrame([asdict(check) for check in readiness_checks(repo_root)])
        display(checks[["name", "passed", "detail"]])
        checks.loc[~checks["passed"], ["name", "how_to_fix"]]
        """
    ),
]


if __name__ == "__main__":
    print(save(CELLS, NOTEBOOK_FILE))
