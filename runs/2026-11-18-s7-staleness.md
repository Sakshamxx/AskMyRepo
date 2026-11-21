# S7 planted failure: staleness (rehearsed 2026-11-18)

```
Pasted: ShopLite main (2.0) and the warehouse, 46,614 tokens
Q: In ShopLite, how many coupons can a customer use on one order?
A: Based on the repository, a customer can use **only one coupon** per order. 

This is specified in:
* `shoplite/CHANGELOG.md` under version `1.1.0` ("Coupons (percent or flat), one per order...")
* `shoplite/docs/api.md` ("Apply a coupon `code` (one per order)")
* `shoplite/shoplite/services/coupons.py` (`MAX_COUPONS_PER_ORDER = 1`, and the check `if len(cart.coupons) >= MAX_COUPONS_PER_ORDER: raise Conflict("only one coupon can be used per order", code="SL-CPN-409")`)

File used: `shoplite/shoplite/services/coupons.py` (as well as `shoplite/docs/api.md` and `shoplite/CHANGELOG.md`).
(6.2 s, $0.0145)

The code after feat/coupon-stacking (git diff main origin/feat/coupon-stacking -- shoplite/services/coupons.py):
-MAX_COUPONS_PER_ORDER = 1
+MAX_COUPONS_PER_ORDER = 2  # at most one percent and one flat coupon
```
