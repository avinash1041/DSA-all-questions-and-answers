# 🐍 THE COMPLETE DJANGO & FASTAPI SENIOR DEVELOPER INTERVIEW GUIDE
## From Basic → Intermediate → Advanced → Expert Level
### Production-Ready | Real-World Examples | 15+ Years of Industry Knowledge

---

> **How to use this guide:**
> Read it top to bottom once. Then revise section-by-section.
> Every concept is explained like teaching a child — simple language, real examples, then technical depth.
> After this guide, you will walk into any senior backend interview with full confidence.

---

# TABLE OF CONTENTS

1. SECTION 1 — Python Basics to Advanced
2. SECTION 2 — Django Complete Interview Guide
3. SECTION 3 — FastAPI Complete Interview Guide
4. SECTION 4 — Celery Complete Guide
5. SECTION 5 — Database Interview Questions
6. SECTION 6 — System Design & Architecture
7. SECTION 7 — Production & Deployment
8. SECTION 8 — Coding Standards & Best Practices
9. SECTION 9 — Real Interview Preparation

---

# ═══════════════════════════════════════════════════
# SECTION 1 — PYTHON BASICS TO ADVANCED
# ═══════════════════════════════════════════════════

---

## 1.1 VARIABLES & DATA TYPES

### Q: What are variables and data types in Python?

**Simple Explanation (like explaining to a 5-year-old):**
Think of a variable like a box. You put something inside the box and give the box a name.
"My box named `age` has the number `25` inside."

**Data types = what kind of thing is inside the box.**
- Numbers → `int`, `float`
- Text → `str`
- True/False → `bool`
- Nothing → `None`
- List of things → `list`
- Fixed list → `tuple`
- Unique items → `set`
- Key-value pairs → `dict`

```python
# Variables
name = "Rahul"          # str
age = 25                # int
salary = 55000.50       # float
is_active = True        # bool
nothing = None          # NoneType

# Checking type
print(type(name))       # <class 'str'>
print(type(age))        # <class 'int'>
```

**Real-world example:**
In a Django model, each field maps to a data type:
```python
class Employee(models.Model):
    name = models.CharField(max_length=100)   # str
    age = models.IntegerField()               # int
    salary = models.DecimalField(...)         # float
    is_active = models.BooleanField()         # bool
```

**Common mistakes:**
- Using `=` (assignment) vs `==` (comparison) — classic bug
- `0`, `""`, `[]`, `None`, `False` are all FALSY in Python
- `int("25abc")` throws ValueError — always validate before conversion

---

## 1.2 MUTABLE vs IMMUTABLE

### Q: What is mutable and immutable? Why does it matter?

**Simple Explanation:**
- **Immutable** = Once created, you CANNOT change it. Like a stone engraving.
- **Mutable** = You CAN change it after creation. Like writing on a whiteboard.

| Type      | Mutable? |
|-----------|----------|
| int       | NO       |
| float     | NO       |
| str       | NO       |
| tuple     | NO       |
| list      | YES      |
| dict      | YES      |
| set       | YES      |

```python
# Immutable — str
name = "Rahul"
name[0] = "K"      # ERROR! TypeError: 'str' object does not support item assignment

# Mutable — list
fruits = ["apple", "banana"]
fruits[0] = "mango"   # Works fine!
print(fruits)          # ['mango', 'banana']

# The trick with immutable types
a = "hello"
b = a
a = "world"
print(b)  # Still "hello" — b was not affected
```

**Why this matters in interviews:**
```python
# DANGEROUS — mutable default argument (common trap!)
def add_item(item, my_list=[]):   # DON'T DO THIS
    my_list.append(item)
    return my_list

print(add_item("a"))   # ['a']
print(add_item("b"))   # ['a', 'b']  ← Bug! List persists between calls

# CORRECT way:
def add_item(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list
```

**Real-world production scenario:**
In FastAPI, Pydantic models are immutable by default. If you need to modify data, you use `.copy(update={...})` instead of direct mutation. This prevents bugs in async code where multiple requests might share state.

---

## 1.3 LIST vs TUPLE vs SET vs DICTIONARY

### Q: When do you use each one?

**Simple Analogy:**
- **List** = Shopping cart (ordered, can add/remove, can repeat items)
- **Tuple** = Coordinates (fixed, ordered, never changes: latitude, longitude)
- **Set** = Unique raffle tickets (no duplicates, no order)
- **Dict** = Phone book (name → number mapping)

```python
# LIST — ordered, mutable, allows duplicates
cart = ["apple", "banana", "apple"]
cart.append("orange")
cart.remove("banana")

# TUPLE — ordered, immutable
coordinates = (19.0760, 72.8777)  # Mumbai lat, long
# coordinates[0] = 20  # ERROR!

# SET — unordered, unique, fast membership check
unique_users = {"user1", "user2", "user1"}
print(unique_users)  # {'user1', 'user2'} — duplicate removed

# Check membership: O(1) for set, O(n) for list
"user1" in unique_users   # Very fast!

# DICT — key-value pairs
employee = {
    "name": "Rahul",
    "age": 25,
    "skills": ["Python", "Django"]
}
print(employee["name"])   # Rahul
employee.get("salary", 0) # Returns 0 if key not found (safe way)
```

**Performance comparison (important for interviews):**

| Operation          | List    | Set     | Dict    |
|--------------------|---------|---------|---------|
| Search (in)        | O(n)    | O(1)    | O(1)    |
| Insert             | O(1)    | O(1)    | O(1)    |
| Delete             | O(n)    | O(1)    | O(1)    |

**Real-world example:**
```python
# In Django views — converting QuerySet to set for fast lookups
active_user_ids = set(User.objects.filter(is_active=True).values_list('id', flat=True))

# Now checking if a user is active is O(1) instead of O(n)
if user_id in active_user_ids:
    process_request(user_id)
```

---

## 1.4 OOP CONCEPTS

### Q: Explain OOP concepts with real-world examples.

**Simple Explanation:**
OOP = Object-Oriented Programming. We model real-world things as objects.
Think of a **class** as a blueprint (cookie cutter) and an **object** as the actual cookie.

### Encapsulation
**= Hiding internal details. Like a TV remote — you press buttons, you don't need to know the circuit.**

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # Private (double underscore = name mangling)
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
    
    def withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance -= amount
        else:
            raise ValueError("Insufficient funds")
    
    def get_balance(self):  # Controlled access via property
        return self.__balance

account = BankAccount("Rahul", 1000)
account.deposit(500)
print(account.get_balance())  # 1500
# print(account.__balance)    # AttributeError — protected!
```

### Inheritance
**= Child gets properties from parent. Like you inherited your dad's eyes.**

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def breathe(self):
        return f"{self.name} is breathing"

class Dog(Animal):
    def speak(self):   # Overriding parent method
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog("Bruno")
cat = Cat("Whiskers")
print(dog.speak())    # Woof!
print(cat.breathe())  # Whiskers is breathing (inherited)
```

**Django Real-world Example:**
```python
# Django itself uses inheritance everywhere
class UserProfile(AbstractUser):  # Inheriting from Django's User
    phone = models.CharField(max_length=15)
    avatar = models.ImageField(upload_to='avatars/')
    bio = models.TextField(blank=True)
```

### Polymorphism
**= Same action, different behavior. Like "area" for circle vs rectangle.**

```python
class Shape:
    def area(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

# Polymorphism in action
shapes = [Circle(5), Rectangle(4, 6)]
for shape in shapes:
    print(shape.area())  # Each calls its own area()
```

### Abstraction
**= Show only what's needed, hide complexity. Like driving a car — you use the steering wheel, you don't need to know the engine.**

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    
    @abstractmethod
    def process_payment(self, amount):
        pass
    
    @abstractmethod
    def refund(self, transaction_id):
        pass
    
    def log_transaction(self, amount):  # Concrete method
        print(f"Transaction logged: {amount}")

class StripeProcessor(PaymentProcessor):
    def process_payment(self, amount):
        # Stripe-specific logic
        return {"status": "success", "provider": "stripe"}
    
    def refund(self, transaction_id):
        return {"refunded": True}

class RazorpayProcessor(PaymentProcessor):
    def process_payment(self, amount):
        # Razorpay-specific logic
        return {"status": "success", "provider": "razorpay"}
    
    def refund(self, transaction_id):
        return {"refunded": True}

# In production, you'd inject which processor to use
processor = StripeProcessor()
processor.process_payment(1000)
```

---

## 1.5 DECORATORS

### Q: What is a decorator? Explain with examples.

**Simple Explanation:**
A decorator is a wrapper around a function. Like wrapping a gift — the gift (function) is inside, the wrapper (decorator) adds extra packaging.

```python
# Basic decorator structure
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before the function runs")
        result = func(*args, **kwargs)
        print("After the function runs")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before the function runs
# Hello!
# After the function runs
```

**Real-world production decorators:**

```python
import time
import functools
import logging

logger = logging.getLogger(__name__)

# 1. Timing decorator — measure performance
def timer(func):
    @functools.wraps(func)  # Preserves original function metadata
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

# 2. Retry decorator — for unreliable operations
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

# 3. Cache decorator
def cache(func):
    memo = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper

# Usage
@timer
@retry(max_attempts=3, delay=2)
def call_external_api(url):
    import requests
    return requests.get(url).json()

@cache
def expensive_calculation(n):
    return sum(range(n))
```

**Django/FastAPI Decorator Examples:**
```python
# Django — login_required is a decorator
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

# FastAPI — Depends() is dependency injection (decorator pattern)
from fastapi import FastAPI, Depends

app = FastAPI()

def verify_token(token: str):
    if token != "secret":
        raise HTTPException(status_code=401)
    return token

@app.get("/protected")
def protected_route(token: str = Depends(verify_token)):
    return {"message": "Access granted"}
```

**Common Mistakes:**
- Forgetting `@functools.wraps(func)` — causes `func.__name__` to show `wrapper` instead
- Decorators with arguments need 3 levels of functions (decorator factory)
- Order of decorators matters — they apply bottom to top

---

## 1.6 GENERATORS & YIELD

### Q: What are generators? Why are they better than lists for large data?

**Simple Explanation:**
Imagine you have 1 million cookies to give out.
- **List approach**: Bake ALL 1 million cookies first, put them in a giant room, then give one by one. (Wastes memory)
- **Generator approach**: Bake ONE cookie, give it, bake the next one when asked. (Saves memory)

```python
# LIST — loads everything into memory
def get_all_numbers(n):
    return [x * 2 for x in range(n)]

numbers = get_all_numbers(1_000_000)  # Stores 1M items in RAM!

# GENERATOR — produces one at a time
def generate_numbers(n):
    for x in range(n):
        yield x * 2  # "yield" = pause here, give value, resume when asked

gen = generate_numbers(1_000_000)  # Stores NOTHING yet
next(gen)   # 0
next(gen)   # 2
next(gen)   # 4

# More Pythonic
for number in generate_numbers(1_000_000):
    process(number)   # Process one at a time, O(1) memory
```

**Real-world production example — reading large CSV files:**
```python
# BAD — reads entire file into memory
def read_csv_bad(filepath):
    with open(filepath) as f:
        return f.readlines()  # 10GB file = 10GB RAM usage!

# GOOD — generator processes line by line
def read_csv_good(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip().split(',')

# Usage in Django management command for bulk imports
def process_large_import(filepath):
    for row in read_csv_good(filepath):
        Employee.objects.create(
            name=row[0],
            email=row[1],
            department=row[2]
        )
```

**Generator expressions:**
```python
# List comprehension — creates full list in memory
squares_list = [x**2 for x in range(10000)]

# Generator expression — lazy, only computes when needed
squares_gen = (x**2 for x in range(10000))

# Generator chaining
import itertools

def read_logs(filepath):
    with open(filepath) as f:
        yield from f

def filter_errors(lines):
    for line in lines:
        if "ERROR" in line:
            yield line

def parse_error(line):
    return {"timestamp": line[:19], "message": line[20:]}

# Pipeline — processes one line at a time through entire pipeline
error_logs = (parse_error(line) for line in filter_errors(read_logs("app.log")))
```

---

## 1.7 ASYNC PROGRAMMING, ASYNC/AWAIT, EVENT LOOP

### Q: Explain async programming in Python simply.

**Simple Explanation:**
Imagine a waiter at a restaurant:
- **Synchronous (old way)**: Waiter takes order from table 1, goes to kitchen, WAITS there for food, comes back, then takes order from table 2. Very slow!
- **Asynchronous (new way)**: Waiter takes order from table 1, goes to kitchen, gives order, then goes to table 2 while food is being cooked. When food is ready, comes back to table 1.

The waiter is the **event loop**. The waiting-for-food is **I/O (network, database calls)**.

```python
import asyncio
import aiohttp

# SYNCHRONOUS — slow, blocks
import requests
def fetch_sync(url):
    response = requests.get(url)   # BLOCKS here until response arrives
    return response.json()

# ASYNCHRONOUS — fast, doesn't block
async def fetch_async(session, url):
    async with session.get(url) as response:   # Non-blocking!
        return await response.json()

async def main():
    urls = [
        "https://api.github.com/users/user1",
        "https://api.github.com/users/user2",
        "https://api.github.com/users/user3",
    ]
    
    async with aiohttp.ClientSession() as session:
        # Run all requests CONCURRENTLY
        tasks = [fetch_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # All run at same time!
    
    return results

# Run the async function
asyncio.run(main())
```

**Event Loop — the boss that manages everything:**
```
Event Loop
    │
    ├── Task 1: Fetch DB  ──────► Waiting... (non-blocking)
    │                                         │
    ├── Task 2: Fetch API ──────► Waiting...  │
    │                                         │
    └── Task 3: Read File ──────► Done! ◄─────┘
                                 Processing Task 3
                                 While waiting for Task 1, 2...
```

**FastAPI uses async deeply:**
```python
from fastapi import FastAPI
from databases import Database

app = FastAPI()
database = Database("postgresql://user:pass@localhost/db")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # async query — doesn't block other requests!
    query = "SELECT * FROM users WHERE id = :id"
    return await database.fetch_one(query=query, values={"id": user_id})
```

**async vs sync — when to use which:**
| Use Case                    | Use Async? |
|-----------------------------|------------|
| Database queries            | YES        |
| External API calls          | YES        |
| File I/O                    | YES        |
| CPU-heavy computation       | NO (use multiprocessing) |
| Simple string operations    | NO (overkill) |

---

## 1.8 GIL (Global Interpreter Lock)

### Q: What is the GIL? Why does it matter?

**Simple Explanation:**
Imagine a kitchen with one knife. Multiple chefs (threads) want to chop vegetables (run Python code). The GIL is a rule: **only one chef can use the knife at a time**.

This means:
- **Threading in Python** = Multiple chefs, but only ONE can work at any moment (for CPU tasks)
- **For I/O tasks** (waiting for network, files) → GIL is released → threads work truly in parallel
- **Multiprocessing** = Multiple kitchens, each with their own knife → TRUE parallelism

```python
import threading
import multiprocessing
import time

# CPU-BOUND task — GIL limits threading
def cpu_task(n):
    return sum(range(n))

# Threading (GIL blocks true parallelism for CPU tasks)
start = time.time()
threads = [threading.Thread(target=cpu_task, args=(10_000_000,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"Threading: {time.time() - start:.2f}s")  # Not much faster

# Multiprocessing (bypasses GIL — true parallelism)
start = time.time()
with multiprocessing.Pool(4) as pool:
    pool.map(cpu_task, [10_000_000] * 4)
print(f"Multiprocessing: {time.time() - start:.2f}s")  # ~4x faster!

# I/O-BOUND task — threading IS effective
def io_task(url):
    import urllib.request
    urllib.request.urlopen(url)

# For I/O, threading works great because GIL is released during I/O waits
```

**In production:**
- **Gunicorn with multiple workers** = multiprocessing (each worker has its own GIL)
- **Celery** = separate processes for CPU tasks
- **FastAPI with async** = single thread handles thousands of I/O-bound requests

---

## 1.9 MEMORY MANAGEMENT

### Q: How does Python manage memory? What is garbage collection?

**Simple Explanation:**
Python automatically cleans up memory — like a restaurant cleaning tables after customers leave. You don't need to do it manually (unlike C/C++).

Python uses **Reference Counting** + **Cyclic Garbage Collector**.

```python
import sys
import gc

# Reference counting
x = [1, 2, 3]
print(sys.getrefcount(x))  # 2 (one for x, one for the function call)

y = x  # Now two variables point to same list
print(sys.getrefcount(x))  # 3

del y  # Decrease count
print(sys.getrefcount(x))  # 2

del x  # Count goes to 0 → Python frees the memory!

# Circular references (can cause memory leaks)
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

a = Node(1)
b = Node(2)
a.next = b
b.next = a   # Circular reference! a → b → a → b...
# del a, del b — ref count never reaches 0!
# Python's cyclic GC handles this

gc.collect()  # Force garbage collection

# Memory optimization tips
# Use __slots__ to reduce memory for many objects
class OptimizedPoint:
    __slots__ = ['x', 'y']  # Prevents __dict__ creation, saves ~40% memory
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Regular class — each instance has a __dict__ (overhead)
# With __slots__ — no __dict__, fixed attributes, much more memory efficient
```

---

## 1.10 DEEP COPY vs SHALLOW COPY

### Q: What's the difference? Give examples.

**Simple Explanation:**
- **Shallow copy**: Copy the box, but both boxes share the same items inside. Change an item in one box → affects both!
- **Deep copy**: Copy EVERYTHING — the box AND all items inside. Completely independent copies.

```python
import copy

original = {"name": "Rahul", "skills": ["Python", "Django"]}

# Shallow copy
shallow = original.copy()  # or copy.copy(original)
shallow["name"] = "Amit"       # Changes only in shallow copy ✓
shallow["skills"].append("FastAPI")  # CHANGES ORIGINAL TOO! ✗

print(original["name"])    # "Rahul"  ← name unchanged
print(original["skills"])  # ["Python", "Django", "FastAPI"]  ← CHANGED!

# Deep copy
original2 = {"name": "Rahul", "skills": ["Python", "Django"]}
deep = copy.deepcopy(original2)
deep["skills"].append("FastAPI")

print(original2["skills"])  # ["Python", "Django"]  ← NOT changed ✓
print(deep["skills"])       # ["Python", "Django", "FastAPI"]
```

**Real-world scenario:**
```python
# In Django, when modifying queryset results
from copy import deepcopy

users = User.objects.filter(is_active=True)
users_copy = deepcopy(users)  # Fully independent copy for modification
```

---

## 1.11 ERROR HANDLING & LOGGING

### Q: How do you handle errors in production Python code?

```python
import logging
import traceback
from typing import Optional

# Setting up production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Proper exception hierarchy
class AppError(Exception):
    """Base exception for our app"""
    pass

class DatabaseError(AppError):
    """Database-related errors"""
    def __init__(self, message, query=None):
        self.query = query
        super().__init__(message)

class ValidationError(AppError):
    """Input validation errors"""
    pass

# Production-ready error handling
def process_payment(user_id: int, amount: float) -> dict:
    try:
        # Validate
        if amount <= 0:
            raise ValidationError(f"Invalid amount: {amount}")
        
        # Database operation
        user = get_user(user_id)
        if not user:
            raise DatabaseError(f"User {user_id} not found")
        
        # Payment processing
        result = payment_gateway.charge(user, amount)
        logger.info(f"Payment successful: user={user_id}, amount={amount}")
        return {"status": "success", "transaction_id": result.id}
    
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return {"status": "error", "code": "VALIDATION_ERROR", "message": str(e)}
    
    except DatabaseError as e:
        logger.error(f"Database error: {e}, query={e.query}")
        return {"status": "error", "code": "DB_ERROR", "message": "Database unavailable"}
    
    except Exception as e:
        logger.critical(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "code": "INTERNAL_ERROR", "message": "Internal server error"}
    
    finally:
        logger.debug(f"process_payment completed for user_id={user_id}")
```

---

## 1.12 CONTEXT MANAGERS

### Q: What is a context manager? When do you use it?

**Simple Explanation:**
A context manager handles setup and cleanup automatically. Like a valet parking service — they take your car when you arrive (setup), and return it when you leave (cleanup). You don't worry about it.

```python
# The classic example — file handling
# WITHOUT context manager (BAD — file might not close on error)
f = open("data.txt", "r")
data = f.read()
f.close()  # What if an error occurs before this?

# WITH context manager (GOOD — always closes, even on error)
with open("data.txt", "r") as f:
    data = f.read()
# File is automatically closed here

# Creating your own context manager
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Context manager to time code blocks"""
    start = time.perf_counter()
    try:
        yield  # Control goes to the 'with' block here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.4f}s")

# Usage
with timer("Database Query"):
    results = User.objects.filter(is_active=True)

# Database transaction context manager
@contextmanager
def db_transaction(db_connection):
    try:
        yield db_connection
        db_connection.commit()
        print("Transaction committed")
    except Exception:
        db_connection.rollback()
        print("Transaction rolled back")
        raise

# Class-based context manager
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    def __enter__(self):
        self.connection = create_connection(self.host, self.port)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
        return False  # Don't suppress exceptions

with DatabaseConnection("localhost", 5432) as conn:
    conn.execute("SELECT * FROM users")
```

---

## 1.13 SOLID PRINCIPLES

### Q: Explain SOLID principles with Python examples.

**Simple Explanation:**
SOLID = 5 design rules to write code that's easy to maintain, extend, and test.

**S — Single Responsibility Principle**
= Each class/function does ONE thing only.

```python
# BAD — class does too many things
class UserManager:
    def create_user(self, data): ...
    def send_welcome_email(self, user): ...  # Email is separate concern!
    def save_to_database(self, user): ...     # DB is separate concern!
    def generate_pdf_report(self, user): ...  # Report is separate concern!

# GOOD — each class has one job
class UserRepository:
    def save(self, user): ...
    def find_by_id(self, user_id): ...

class EmailService:
    def send_welcome_email(self, user): ...

class ReportService:
    def generate_user_report(self, user): ...

class UserService:
    def __init__(self, repo, email_service):
        self.repo = repo
        self.email_service = email_service
    
    def create_user(self, data):
        user = User(**data)
        self.repo.save(user)
        self.email_service.send_welcome_email(user)
        return user
```

**O — Open/Closed Principle**
= Open for extension, closed for modification.

```python
# BAD — need to modify existing code to add new payment method
class PaymentProcessor:
    def process(self, payment_type, amount):
        if payment_type == "card":
            # card logic
        elif payment_type == "upi":
            # upi logic
        elif payment_type == "crypto":  # Had to modify class!
            # crypto logic

# GOOD — extend without modifying
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount): pass

class CardPayment(PaymentMethod):
    def process(self, amount):
        return f"Card charged: ₹{amount}"

class UPIPayment(PaymentMethod):
    def process(self, amount):
        return f"UPI transferred: ₹{amount}"

class CryptoPayment(PaymentMethod):  # NEW class, old code untouched!
    def process(self, amount):
        return f"Crypto sent: {amount}"

class PaymentProcessor:
    def process(self, payment: PaymentMethod, amount: float):
        return payment.process(amount)
```

**L — Liskov Substitution Principle**
= Child class can replace parent class without breaking things.

```python
class Bird:
    def fly(self):
        return "Flying"

class Eagle(Bird):   # Works fine
    def fly(self):
        return "Eagle soaring"

class Penguin(Bird):  # VIOLATES LSP!
    def fly(self):
        raise Exception("Penguins can't fly!")  # Breaks parent contract

# Fix: Redesign hierarchy
class Bird:
    def move(self): pass

class FlyingBird(Bird):
    def fly(self): return "Flying"

class SwimmingBird(Bird):
    def swim(self): return "Swimming"

class Eagle(FlyingBird): pass
class Penguin(SwimmingBird): pass
```

**I — Interface Segregation Principle**
= Don't force classes to implement methods they don't need.

```python
# BAD — fat interface
class Worker(ABC):
    @abstractmethod
    def work(self): pass
    
    @abstractmethod
    def eat(self): pass   # Robots don't eat!
    
    @abstractmethod
    def sleep(self): pass  # Robots don't sleep!

# GOOD — segregated interfaces
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Human(Workable, Eatable):
    def work(self): return "Working"
    def eat(self): return "Eating"

class Robot(Workable):
    def work(self): return "Computing"
    # No eat() needed!
```

**D — Dependency Inversion Principle**
= High-level modules should not depend on low-level modules. Both should depend on abstractions.

```python
# BAD — UserService depends directly on MySQL
class MySQLDatabase:
    def query(self, sql): ...

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # Tight coupling!
    
    def get_user(self, user_id):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# GOOD — depend on abstraction
class Database(ABC):
    @abstractmethod
    def query(self, sql): pass

class MySQLDatabase(Database):
    def query(self, sql): ...

class PostgreSQLDatabase(Database):
    def query(self, sql): ...

class UserService:
    def __init__(self, db: Database):  # Depends on abstraction, not implementation
        self.db = db
    
    def get_user(self, user_id):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# FastAPI / Django use DI heavily
user_service = UserService(PostgreSQLDatabase())  # Easy to swap!
```

---

## 1.14 DESIGN PATTERNS

### Q: What design patterns do you use in Python/Django/FastAPI?

**Repository Pattern:**
```python
from abc import ABC, abstractmethod
from typing import List, Optional

class UserRepository(ABC):
    @abstractmethod
    def get_by_id(self, user_id: int) -> Optional[dict]: pass
    
    @abstractmethod
    def get_all(self) -> List[dict]: pass
    
    @abstractmethod
    def save(self, user: dict) -> dict: pass

class DjangoUserRepository(UserRepository):
    def get_by_id(self, user_id: int):
        return User.objects.filter(id=user_id).first()
    
    def get_all(self):
        return list(User.objects.all())
    
    def save(self, user_data: dict):
        return User.objects.create(**user_data)

class InMemoryUserRepository(UserRepository):  # For testing!
    def __init__(self):
        self._users = {}
    
    def get_by_id(self, user_id):
        return self._users.get(user_id)
    
    def get_all(self):
        return list(self._users.values())
    
    def save(self, user):
        self._users[user['id']] = user
        return user
```

**Singleton Pattern:**
```python
class DatabasePool:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, max_connections=10):
        if not hasattr(self, '_initialized'):
            self.max_connections = max_connections
            self.connections = []
            self._initialized = True

# Always returns same instance
pool1 = DatabasePool(10)
pool2 = DatabasePool(10)
print(pool1 is pool2)  # True
```

**Factory Pattern:**
```python
class NotificationFactory:
    @staticmethod
    def create(notification_type: str):
        factories = {
            "email": EmailNotification,
            "sms": SMSNotification,
            "push": PushNotification,
        }
        if notification_type not in factories:
            raise ValueError(f"Unknown notification type: {notification_type}")
        return factories[notification_type]()

notification = NotificationFactory.create("email")
notification.send("Hello!")
```

---

# ═══════════════════════════════════════════════════
# SECTION 2 — DJANGO COMPLETE INTERVIEW GUIDE
# ═══════════════════════════════════════════════════

---

## 2.1 WHAT IS DJANGO & WHY DJANGO?

### Q: What is Django? Why would you choose it?

**Simple Explanation:**
Django is like a fully furnished apartment — everything you need is already there: kitchen (ORM), bedroom (templates), security system (auth), plumbing (routing). You just move in and start living.

Flask is like an empty room — you bring your own furniture.

**Django's advantages:**
- **Batteries included**: ORM, Auth, Admin, Forms, Caching, Email — all built-in
- **DRY** (Don't Repeat Yourself): Write code once
- **Security**: CSRF, XSS, SQL injection protection built-in
- **Scalable**: Used by Instagram, Pinterest, Disqus, NASA
- **Admin panel**: Auto-generated admin UI from models
- **ORM**: Powerful database abstraction

---

## 2.2 DJANGO MTV ARCHITECTURE

### Q: Explain Django's architecture.

**Simple Explanation:**
Django uses MTV, similar to MVC:
- **Model** = Database layer (what data looks like)
- **Template** = Presentation layer (what user sees)  
- **View** = Logic layer (what happens)

```
Browser → URL Router → View → Model (DB) → View → Template → Browser

Request Flow:
1. User types URL: /users/25/
2. urls.py matches pattern: path('users/<int:pk>/', UserDetailView.as_view())
3. View runs: gets user from DB via Model
4. View passes data to Template (or returns JSON for API)
5. Response sent back to browser
```

**Django Request-Response Cycle (detailed):**
```
HTTP Request
     │
     ▼
WSGI Server (Gunicorn)
     │
     ▼
Django Middleware Stack (top to bottom)
  - SecurityMiddleware
  - SessionMiddleware
  - CommonMiddleware
  - CsrfViewMiddleware
  - AuthenticationMiddleware
  - MessageMiddleware
     │
     ▼
URL Router (urls.py)
     │
     ▼
View Function/Class
     │
     ├── Model (ORM → Database)
     │
     ├── Cache Check
     │
     ▼
Response Object
     │
     ▼
Middleware Stack (bottom to top)
     │
     ▼
HTTP Response
```

---

## 2.3 PRODUCTION-READY DJANGO FOLDER STRUCTURE

```
my_project/
│
├── config/                    # Project configuration
│   ├── __init__.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py           # Common settings
│   │   ├── development.py    # Dev settings
│   │   ├── production.py     # Production settings
│   │   └── testing.py        # Test settings
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── apps/                      # All Django apps
│   ├── users/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── api/
│   │   │   ├── views.py
│   │   │   ├── serializers.py
│   │   │   ├── urls.py
│   │   │   └── permissions.py
│   │   ├── models.py
│   │   ├── services.py       # Business logic
│   │   ├── repositories.py   # DB operations
│   │   ├── tasks.py          # Celery tasks
│   │   ├── signals.py
│   │   └── tests/
│   │       ├── test_models.py
│   │       ├── test_views.py
│   │       └── test_services.py
│   │
│   ├── products/
│   └── orders/
│
├── common/                    # Shared utilities
│   ├── exceptions.py
│   ├── permissions.py
│   ├── pagination.py
│   ├── mixins.py
│   └── utils.py
│
├── static/
├── media/
├── templates/
├── requirements/
│   ├── base.txt
│   ├── development.txt
│   └── production.txt
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
│
├── .env
├── .env.example
├── manage.py
└── README.md
```

---

## 2.4 DJANGO ORM — DEEP DIVE

### Q: Explain Django ORM and its advanced features.

**Simple Explanation:**
ORM = Object Relational Mapper. Instead of writing raw SQL, you write Python code, and Django converts it to SQL automatically.

```python
# Instead of:
# SELECT * FROM users WHERE is_active = True AND age > 18 LIMIT 10;

# You write:
users = User.objects.filter(is_active=True, age__gt=18)[:10]
```

### QuerySet — Lazy Evaluation

```python
# QuerySets are LAZY — they don't hit the DB until evaluated!
users = User.objects.filter(is_active=True)  # NO DB QUERY YET
users = users.filter(age__gt=18)             # STILL NO DB QUERY
users = users.order_by('-created_at')        # STILL NO DB QUERY

# DB query happens when:
list(users)          # ← HERE
for user in users:   # ← HERE
users[0]             # ← HERE
users.count()        # ← HERE
```

### select_related vs prefetch_related

**This is one of the MOST asked Django ORM interview questions!**

**Simple Explanation:**
- `select_related` = One SQL JOIN query (for ForeignKey/OneToOne)
- `prefetch_related` = Two SQL queries, Python joins them (for ManyToMany/reverse ForeignKey)

```python
# MODELS
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)  # ForeignKey

class Tag(models.Model):
    name = models.CharField(max_length=50)

class Article(models.Model):
    title = models.CharField(max_length=200)
    tags = models.ManyToManyField(Tag)  # ManyToMany

# BAD — N+1 Query Problem!
books = Book.objects.all()  # 1 query
for book in books:
    print(book.author.name)  # N queries! (one per book)
# Total: 1 + N queries = VERY SLOW for large datasets

# GOOD — select_related (for ForeignKey)
books = Book.objects.select_related('author').all()  # 1 JOIN query
for book in books:
    print(book.author.name)  # NO extra query! Author already loaded
# SQL: SELECT books.*, authors.* FROM books INNER JOIN authors ON ...

# GOOD — prefetch_related (for ManyToMany / reverse FK)
articles = Article.objects.prefetch_related('tags').all()  # 2 queries total
for article in articles:
    print(article.tags.all())  # NO extra query!
# Query 1: SELECT * FROM articles
# Query 2: SELECT * FROM tags WHERE article_id IN (1,2,3,...)
# Python joins them in memory

# Complex example
from django.db.models import Prefetch

books = Book.objects.select_related(
    'author',
    'author__profile'  # Can chain!
).prefetch_related(
    Prefetch(
        'reviews',
        queryset=Review.objects.filter(rating__gte=4),
        to_attr='good_reviews'  # Store in custom attribute
    )
)
```

### annotate() and aggregate()

```python
from django.db.models import Count, Sum, Avg, Max, Min, F

# aggregate — returns a single dict
stats = Order.objects.aggregate(
    total_orders=Count('id'),
    total_revenue=Sum('amount'),
    avg_order=Avg('amount'),
    max_order=Max('amount')
)
# {'total_orders': 1500, 'total_revenue': 500000, ...}

# annotate — adds calculated field to each object
from django.db.models import Count

authors = Author.objects.annotate(
    book_count=Count('book')  # Add book_count field to each author
).filter(book_count__gt=5)   # Authors with more than 5 books

# F expressions — reference field values
from django.db.models import F

# Increment all prices by 10%
Product.objects.update(price=F('price') * 1.1)

# Compare two fields
orders = Order.objects.filter(
    actual_delivery__gt=F('expected_delivery')  # Late orders!
)

# Q objects — complex queries
from django.db.models import Q

# OR query
users = User.objects.filter(
    Q(email__contains='@gmail.com') | Q(email__contains='@yahoo.com')
)

# Complex AND/OR combination
products = Product.objects.filter(
    Q(price__lt=1000) & Q(in_stock=True) |
    Q(is_featured=True)
)
```

### Database Transactions

```python
from django.db import transaction

# Atomic transaction — all or nothing
@transaction.atomic
def transfer_money(from_account_id, to_account_id, amount):
    from_account = Account.objects.select_for_update().get(id=from_account_id)
    to_account = Account.objects.select_for_update().get(id=to_account_id)
    
    if from_account.balance < amount:
        raise ValueError("Insufficient balance")
    
    from_account.balance -= amount
    to_account.balance += amount
    
    from_account.save()
    to_account.save()
    
    # If any exception occurs, BOTH saves are rolled back!

# Savepoints
def complex_operation():
    with transaction.atomic():
        do_first_thing()
        
        try:
            with transaction.atomic():  # Savepoint
                do_risky_thing()
        except Exception:
            pass  # Inner transaction rolls back, outer continues
        
        do_final_thing()
```

---

## 2.5 DJANGO REST FRAMEWORK (DRF)

### Q: Explain DRF views hierarchy and when to use each.

**DRF View Hierarchy:**
```
APIView (most control, write everything yourself)
    └── GenericAPIView (adds queryset, serializer_class, etc.)
         └── Mixins (ListModelMixin, CreateModelMixin, etc.)
              └── Generic Views (ListCreateAPIView, etc.)
                   └── ViewSets (Router-friendly, cleanest code)
                        └── ModelViewSet (CRUD for FREE!)
```

```python
from rest_framework import generics, viewsets, status
from rest_framework.response import Response
from rest_framework.views import APIView

# LEVEL 1: APIView — maximum control
class UserListAPIView(APIView):
    def get(self, request):
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# LEVEL 2: Generic Views — less code
class UserListCreateView(generics.ListCreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

# LEVEL 3: ModelViewSet — MINIMUM code, MAXIMUM features
from rest_framework.routers import DefaultRouter
from rest_framework.decorators import action

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    
    def get_queryset(self):
        # Customize based on request
        queryset = super().get_queryset()
        if self.request.user.is_staff:
            return queryset
        return queryset.filter(is_active=True)
    
    def get_serializer_class(self):
        # Different serializer for different actions
        if self.action == 'create':
            return UserCreateSerializer
        elif self.action in ['list', 'retrieve']:
            return UserDetailSerializer
        return UserSerializer
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        user = self.get_object()
        user.is_active = True
        user.save()
        return Response({"status": "activated"})
    
    @action(detail=False, methods=['get'])
    def me(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)

# Register with router
router = DefaultRouter()
router.register('users', UserViewSet, basename='user')
# Automatically creates: GET /users/, POST /users/, GET /users/{id}/, 
# PUT /users/{id}/, PATCH /users/{id}/, DELETE /users/{id}/
# GET /users/{id}/activate/, GET /users/me/
```

### Serializers — Deep Dive

```python
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password

class UserSerializer(serializers.ModelSerializer):
    full_name = serializers.SerializerMethodField()
    password = serializers.CharField(write_only=True, validators=[validate_password])
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'full_name', 'created_at']
        read_only_fields = ['id', 'created_at']
        extra_kwargs = {
            'email': {'required': True},
        }
    
    def get_full_name(self, obj):
        return f"{obj.first_name} {obj.last_name}"
    
    def validate_email(self, value):
        # Field-level validation
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already registered")
        return value.lower()
    
    def validate(self, attrs):
        # Object-level validation
        if attrs.get('password') == attrs.get('username'):
            raise serializers.ValidationError("Password cannot be same as username")
        return attrs
    
    def create(self, validated_data):
        # Override to hash password
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user
    
    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance

# Nested serializers
class AddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Address
        fields = ['street', 'city', 'state', 'pincode']

class UserWithAddressSerializer(serializers.ModelSerializer):
    address = AddressSerializer()
    
    class Meta:
        model = User
        fields = ['id', 'name', 'email', 'address']
    
    def create(self, validated_data):
        address_data = validated_data.pop('address')
        user = User.objects.create(**validated_data)
        Address.objects.create(user=user, **address_data)
        return user
```

### Authentication & Permissions

```python
# settings.py — Default auth & permission classes
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/day',
        'user': '1000/day',
    },
}

# JWT Configuration
from datetime import timedelta
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}

# Custom Permission class
from rest_framework.permissions import BasePermission, SAFE_METHODS

class IsOwnerOrReadOnly(BasePermission):
    """Object owner can edit; others can only read."""
    
    def has_object_permission(self, request, view, obj):
        if request.method in SAFE_METHODS:
            return True  # GET, HEAD, OPTIONS — always allowed
        return obj.owner == request.user

class IsAdminOrReadOnly(BasePermission):
    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True
        return request.user and request.user.is_staff

# Usage in ViewSet
class ProductViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
    
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [AllowAny()]
        return [IsAuthenticated(), IsOwnerOrReadOnly()]
```

---

## 2.6 DJANGO MIDDLEWARE

### Q: What is middleware and how does it work?

**Simple Explanation:**
Middleware = security checkpoint at an airport. Every passenger (request) goes through it before reaching their destination (view), and again when leaving (response).

```python
# How middleware processes requests and responses:
# Request goes DOWN the stack (top to bottom)
# Response goes UP the stack (bottom to top)

# Middleware execution order:
# Request:  M1_in → M2_in → M3_in → View
# Response: M3_out → M2_out → M1_out → Client

# Custom middleware examples
import time
import logging
import uuid

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware:
    """Log every request with timing and correlation ID"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Generate correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        request.correlation_id = correlation_id
        
        start_time = time.perf_counter()
        logger.info(
            f"Request started | {request.method} {request.path} "
            f"| correlation_id={correlation_id} "
            f"| user={getattr(request.user, 'id', 'anonymous')}"
        )
        
        response = self.get_response(request)  # Process the request
        
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Request completed | {request.method} {request.path} "
            f"| status={response.status_code} "
            f"| duration={elapsed:.4f}s "
            f"| correlation_id={correlation_id}"
        )
        
        response['X-Correlation-ID'] = correlation_id
        return response

class MaintenanceModeMiddleware:
    """Return 503 when maintenance mode is on"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        from django.conf import settings
        
        if getattr(settings, 'MAINTENANCE_MODE', False):
            if not request.path.startswith('/admin/'):
                from django.http import JsonResponse
                return JsonResponse(
                    {"error": "Service temporarily unavailable for maintenance"},
                    status=503
                )
        
        return self.get_response(request)

# Register middleware in settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'apps.common.middleware.RequestLoggingMiddleware',  # Your custom one
    'apps.common.middleware.MaintenanceModeMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ...
]
```

---

## 2.7 DJANGO SIGNALS

### Q: What are Django signals? Real-world use cases?

**Simple Explanation:**
Signals = event notifications. Like subscribing to notifications. When something happens (signal sent), your code runs automatically.

Real world: When a new user registers → automatically send welcome email.

```python
from django.db.models.signals import post_save, pre_save, post_delete, pre_delete
from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver, Signal

# 1. Auto-create user profile when user is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:  # Only on creation, not updates
        UserProfile.objects.create(user=instance)
        send_welcome_email.delay(instance.email)  # Celery task

# 2. Log changes before saving
@receiver(pre_save, sender=Order)
def log_order_status_change(sender, instance, **kwargs):
    if instance.pk:  # Existing order (not new)
        try:
            old_order = Order.objects.get(pk=instance.pk)
            if old_order.status != instance.status:
                OrderStatusHistory.objects.create(
                    order=instance,
                    old_status=old_order.status,
                    new_status=instance.status,
                    changed_by=instance.last_updated_by
                )
        except Order.DoesNotExist:
            pass

# 3. Update cache when data changes
@receiver(post_save, sender=Product)
@receiver(post_delete, sender=Product)
def invalidate_product_cache(sender, instance, **kwargs):
    from django.core.cache import cache
    cache.delete(f'product_{instance.id}')
    cache.delete('product_list')

# Custom signals
order_completed = Signal()

def complete_order(order):
    order.status = 'completed'
    order.save()
    order_completed.send(sender=Order, order=order)  # Send custom signal

@receiver(order_completed)
def on_order_completed(sender, order, **kwargs):
    send_order_receipt.delay(order.id)
    update_inventory(order)
    notify_warehouse.delay(order.id)
```

**When NOT to use signals:**
- When the code is tightly related (just put it in the view/service directly)
- Complex business logic (hard to trace/debug signals)
- When you need the result immediately

---

## 2.8 DJANGO CACHING & REDIS

### Q: How do you implement caching in Django?

```python
# settings.py — Redis cache backend
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "SOCKET_CONNECT_TIMEOUT": 5,
            "SOCKET_TIMEOUT": 5,
            "IGNORE_EXCEPTIONS": True,  # Don't crash if Redis is down
        },
        "KEY_PREFIX": "myapp",
        "TIMEOUT": 300,  # Default 5 minutes
    }
}

# 1. View-level caching
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

@cache_page(60 * 15)  # Cache for 15 minutes
def product_list(request):
    products = Product.objects.all()
    return render(request, 'products/list.html', {'products': products})

# For class-based views:
@method_decorator(cache_page(60 * 15), name='dispatch')
class ProductListView(ListView):
    model = Product

# 2. Low-level cache API
from django.core.cache import cache

def get_user_dashboard(user_id):
    cache_key = f"user_dashboard_{user_id}"
    
    # Try to get from cache first
    data = cache.get(cache_key)
    
    if data is None:
        # Cache miss — fetch from DB
        data = {
            "user": UserSerializer(User.objects.get(id=user_id)).data,
            "orders": OrderSerializer(Order.objects.filter(user_id=user_id), many=True).data,
            "stats": calculate_user_stats(user_id),
        }
        cache.set(cache_key, data, timeout=300)  # 5 minutes
    
    return data

# Cache-aside pattern (most common)
def get_product(product_id):
    key = f"product:{product_id}"
    
    product = cache.get(key)
    if product is not None:
        return product
    
    product = Product.objects.select_related('category').get(id=product_id)
    cache.set(key, product, 600)  # 10 minutes
    return product

def update_product(product_id, data):
    product = Product.objects.get(id=product_id)
    for k, v in data.items():
        setattr(product, k, v)
    product.save()
    cache.delete(f"product:{product_id}")  # Invalidate cache!

# 3. Cache with versioning (for cache busting)
cache.set("settings", data, version=2)
cache.get("settings", version=2)
cache.incr_version("settings")  # Bump version, old versions become stale

# 4. Many cache operations at once
cache.set_many({
    "user_1": user_1_data,
    "user_2": user_2_data,
    "user_3": user_3_data,
})
results = cache.get_many(["user_1", "user_2", "user_3"])
cache.delete_many(["user_1", "user_2"])
```

---

## 2.9 CELERY WITH DJANGO

### Q: How do you set up and use Celery with Django?

```python
# celery.py (in config folder)
import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.production')

app = Celery('myproject')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Scheduled tasks
app.conf.beat_schedule = {
    'send-daily-report': {
        'task': 'apps.reports.tasks.send_daily_report',
        'schedule': crontab(hour=8, minute=0),  # Every day at 8 AM
    },
    'cleanup-old-sessions': {
        'task': 'apps.users.tasks.cleanup_old_sessions',
        'schedule': crontab(hour=2, minute=0, day_of_week='monday'),
    },
    'sync-inventory': {
        'task': 'apps.products.tasks.sync_inventory',
        'schedule': 60.0,  # Every 60 seconds
    },
}

# settings.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes

# tasks.py
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # Wait 60s before retry
    autoretry_for=(Exception,),
    retry_backoff=True,       # Exponential backoff
)
def send_order_email(self, order_id: int):
    try:
        order = Order.objects.get(id=order_id)
        send_email(
            to=order.user.email,
            subject=f"Order #{order_id} Confirmed",
            template="order_confirmation.html",
            context={"order": order}
        )
        logger.info(f"Order email sent for order_id={order_id}")
        return {"status": "sent", "order_id": order_id}
    
    except Order.DoesNotExist:
        logger.error(f"Order {order_id} not found")
        return {"status": "failed", "reason": "order not found"}
    
    except SMTPException as exc:
        logger.warning(f"SMTP error for order {order_id}: {exc}")
        raise self.retry(exc=exc)

# Calling tasks
# Async (background)
send_order_email.delay(order_id=123)

# With countdown
send_order_email.apply_async(args=[123], countdown=60)  # Run after 60 seconds

# With ETA
from datetime import datetime, timedelta
eta = datetime.utcnow() + timedelta(minutes=30)
send_order_email.apply_async(args=[123], eta=eta)

# Chaining tasks
from celery import chain, group, chord

workflow = chain(
    validate_order.s(order_id),
    process_payment.s(),
    send_confirmation_email.s(),
)
workflow.delay()
```

---

## 2.10 DJANGO SECURITY

### Q: What security features does Django provide?

**CSRF Protection:**
```python
# Django auto-generates CSRF tokens for forms
# For API endpoints, use CSRFExempt for token-based auth
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  # Only for token-authenticated APIs
def api_endpoint(request):
    pass

# In DRF — authentication handles CSRF differently
# SessionAuthentication enforces CSRF
# TokenAuthentication/JWT does NOT require CSRF
```

**SQL Injection Prevention:**
```python
# SAFE — Django ORM parameterizes queries
users = User.objects.filter(username=username)  # Safe

# DANGEROUS — raw SQL with user input
User.objects.raw(f"SELECT * FROM users WHERE username = '{username}'")  # SQL Injection!

# SAFE raw SQL
User.objects.raw("SELECT * FROM users WHERE username = %s", [username])
```

**XSS Prevention:**
```python
# Django templates auto-escape HTML
{{ user_input }}          # Escaped — safe
{{ user_input|safe }}     # NOT escaped — dangerous!

# DRF serializers escape JSON properly by default

# Content Security Policy header
# Add to middleware or use django-csp package
CONTENT_SECURITY_POLICY = {
    'DIRECTIVES': {
        'default-src': ["'self'"],
        'script-src': ["'self'", "'unsafe-inline'"],  # Avoid unsafe-inline in production!
    }
}
```

**Password Security:**
```python
# settings.py — Strong password validators
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 
     'OPTIONS': {'min_length': 10}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Django uses PBKDF2 by default, but argon2 is better
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',  # Best
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
]
```

---

## 2.11 TOP 50 DJANGO INTERVIEW QUESTIONS

### Q1: What is the N+1 query problem and how do you solve it?
**Answer:** When you fetch N objects and then make 1 additional query for each → N+1 total queries. Solve with `select_related` (FK) and `prefetch_related` (M2M).

### Q2: What's the difference between null=True and blank=True?
```python
# null=True → allows NULL in database
# blank=True → allows empty string in forms/serializers
name = models.CharField(max_length=100, blank=True)  # Form can be empty, DB stores ""
bio = models.TextField(null=True, blank=True)          # Both form empty AND DB NULL allowed
# For CharField/TextField: use blank=True, NOT null=True (avoid dual representation)
# For FK/numeric fields: null=True for optional
```

### Q3: How does Django's authentication work?
```python
# 1. User submits credentials
# 2. Django calls authenticate() — checks username/password
# 3. If valid, login() creates session
# 4. Session ID stored in cookie
# 5. Subsequent requests include session cookie
# 6. Django fetches user from session

# Custom authentication backend
class EmailBackend:
    def authenticate(self, request, username=None, password=None):
        try:
            user = User.objects.get(email=username)
            if user.check_password(password):
                return user
        except User.DoesNotExist:
            return None
    
    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
```

### Q4: What are Django migrations and how do they work?
```bash
# migrations = version control for your database schema
# Each migration file records changes (AddField, DeleteField, etc.)
# Django tracks which migrations have run in django_migrations table

python manage.py makemigrations   # Creates migration files
python manage.py migrate          # Applies migrations to DB
python manage.py showmigrations   # Shows migration status
python manage.py sqlmigrate app 0001  # Shows SQL for a migration
python manage.py migrate app 0001     # Rollback to specific migration
```

### Q5: How do you optimize Django for production?

**Answer (use as talking points in interview):**
1. Use `select_related`/`prefetch_related` to prevent N+1 queries
2. Add database indexes on frequently queried fields
3. Use `only()` and `defer()` to fetch only needed fields
4. Implement caching (Redis) at view/query/model level
5. Use connection pooling (PgBouncer for PostgreSQL)
6. Enable QuerySet `.values()` when you don't need model instances
7. Use `bulk_create()`, `bulk_update()` for batch operations
8. Configure Gunicorn workers properly (2 * CPU + 1)
9. Use CDN for static files
10. Use `CONN_MAX_AGE` for persistent database connections

```python
# CONN_MAX_AGE — reuse DB connections
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'CONN_MAX_AGE': 60,  # Keep connections open for 60 seconds
    }
}

# only() vs defer() — fetch specific fields
users = User.objects.only('id', 'email')  # Fetch only id and email
users = User.objects.defer('bio', 'avatar')  # Fetch everything EXCEPT bio and avatar

# bulk operations
users_to_create = [User(username=f"user{i}", email=f"user{i}@example.com") for i in range(1000)]
User.objects.bulk_create(users_to_create, batch_size=100)

User.objects.filter(is_active=False).update(is_archived=True)  # Bulk update
```

### Q6: What are Django signals and what are the risks?
**Risks:**
- Hard to trace (code runs "magically" from a distance)
- Can cause infinite recursion (signal triggers save → save triggers signal)
- Hard to test
- Can slow down saves if signal handlers are heavy

**Solution:** Use signals sparingly. For business logic, prefer explicit service calls.

### Q7: How do you implement custom user model in Django?
```python
# MUST be done at the START of project. Changing later is very painful!
from django.contrib.auth.models import AbstractUser, AbstractBaseUser

class CustomUser(AbstractUser):
    phone = models.CharField(max_length=15, blank=True)
    email = models.EmailField(unique=True)
    
    USERNAME_FIELD = 'email'  # Login with email instead of username
    REQUIRED_FIELDS = ['username']

# settings.py
AUTH_USER_MODEL = 'users.CustomUser'
```

### Q8: How do you handle file uploads in Django?
```python
class Document(models.Model):
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    image = models.ImageField(upload_to='images/')

# settings.py
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Use S3 for production (not local storage!)
# pip install django-storages boto3
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
AWS_STORAGE_BUCKET_NAME = 'my-bucket'
```

### Q9: Explain Django's caching framework levels.

**Answer (5 levels of caching):**
1. **Per-site cache** — caches entire site
2. **Per-view cache** — `@cache_page` decorator
3. **Template fragment cache** — `{% cache 500 sidebar %}...{% endcache %}`
4. **Low-level cache API** — `cache.get()`, `cache.set()`
5. **Database query cache** — QuerySet caching

### Q10: How do you debug a slow API in Django?

**Production Debugging Steps:**
```python
# 1. Django Debug Toolbar (development only)
# 2. django-silk for profiling
# 3. logging with timing
# 4. Identify slow queries

import logging
import time

logger = logging.getLogger(__name__)

# Add query logging to settings.py (development)
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {'class': 'logging.StreamHandler'},
    },
    'loggers': {
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}

# Django Debug Toolbar shows all queries
# Look for: duplicate queries, missing indexes, slow queries

# 5. Profile with cProfile
import cProfile
import pstats

def profile_view(request):
    with cProfile.Profile() as pr:
        response = expensive_operation()
    
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    return response
```

---

# ═══════════════════════════════════════════════════
# SECTION 3 — FASTAPI COMPLETE INTERVIEW GUIDE
# ═══════════════════════════════════════════════════

---

## 3.1 WHAT IS FASTAPI? WHY FASTAPI?

### Q: Why FastAPI over Django/Flask?

**Simple Explanation:**
- **Django** = Full restaurant with chef, waiter, menu, kitchen — everything
- **Flask** = Kitchen only — you hire your own staff
- **FastAPI** = Modern kitchen designed for speed — async, auto-docs, type hints

**FastAPI's killer features:**
1. **Async by default** — handles thousands of concurrent requests
2. **Automatic docs** — Swagger UI at `/docs`, ReDoc at `/redoc`
3. **Type hints** — Python type hints = automatic validation
4. **Pydantic** — Best-in-class data validation
5. **Speed** — Comparable to Node.js and Go
6. **Dependency injection** — Clean, testable code

---

## 3.2 FASTAPI PRODUCTION FOLDER STRUCTURE

```
fastapi_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app entry point
│   ├── config.py              # Settings using Pydantic BaseSettings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py            # Reusable dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py      # Include all routers
│   │       └── endpoints/
│   │           ├── users.py
│   │           ├── auth.py
│   │           └── products.py
│   │
│   ├── core/
│   │   ├── config.py          # App configuration
│   │   ├── security.py        # JWT, password hashing
│   │   └── exceptions.py      # Custom exceptions
│   │
│   ├── db/
│   │   ├── base.py            # Base model
│   │   ├── session.py         # Database session
│   │   └── init_db.py         # Initial data
│   │
│   ├── models/                # SQLAlchemy models
│   │   ├── user.py
│   │   └── product.py
│   │
│   ├── schemas/               # Pydantic schemas
│   │   ├── user.py
│   │   └── product.py
│   │
│   ├── crud/                  # Database operations
│   │   ├── base.py
│   │   ├── user.py
│   │   └── product.py
│   │
│   ├── services/              # Business logic
│   │   ├── user_service.py
│   │   └── email_service.py
│   │
│   └── middleware/
│       ├── logging.py
│       └── rate_limit.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api/
│   └── test_services/
│
├── alembic/                   # Database migrations
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 3.3 FASTAPI CORE CONCEPTS

```python
from fastapi import FastAPI, Path, Query, Body, Header, Cookie, Depends, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import uvicorn

app = FastAPI(
    title="MyApp API",
    description="Production-ready FastAPI application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# PATH PARAMETERS — part of the URL
@app.get("/users/{user_id}")
async def get_user(
    user_id: int = Path(..., title="User ID", ge=1, description="Must be positive")
):
    return {"user_id": user_id}

# QUERY PARAMETERS — after ? in URL
@app.get("/products")
async def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=0),
    in_stock: bool = Query(True),
):
    # /products?skip=0&limit=10&category=electronics&in_stock=true
    return {"skip": skip, "limit": limit, "category": category}

# REQUEST BODY
class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., regex=r'^[\w.-]+@[\w.-]+\.\w+$')
    age: int = Field(..., ge=18, le=120)
    
    @validator('email')
    def email_must_be_lowercase(cls, v):
        return v.lower()

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # user.name, user.email are validated automatically
    return user

# RESPONSE MODEL
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    
    class Config:
        from_attributes = True  # Allows ORM model → Pydantic (previously orm_mode=True)
```

---

## 3.4 PYDANTIC — DEEP DIVE

### Q: Explain Pydantic v2 features for validation.

```python
from pydantic import (
    BaseModel, Field, validator, model_validator, 
    field_validator, computed_field, ConfigDict
)
from pydantic import EmailStr, HttpUrl, constr, conint
from typing import Optional, List, Literal
from datetime import datetime
from decimal import Decimal

class OrderCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,  # Validate on attribute assignment too
        use_enum_values=True,
    )
    
    # Fields with validation
    customer_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr  # Validates email format automatically
    items: List[OrderItem] = Field(..., min_items=1, max_items=50)
    total_amount: Decimal = Field(..., gt=0, decimal_places=2)
    currency: Literal["INR", "USD", "EUR"] = "INR"
    notes: Optional[str] = Field(None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('customer_name')
    @classmethod
    def name_must_be_letters(cls, v):
        if not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError("Name must contain only letters and spaces")
        return v.title()  # Capitalize
    
    @model_validator(mode='after')
    def check_total_matches_items(self):
        calculated_total = sum(item.price * item.quantity for item in self.items)
        if abs(self.total_amount - calculated_total) > Decimal('0.01'):
            raise ValueError("Total amount doesn't match item sum")
        return self
    
    @computed_field
    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items)

# Settings management with Pydantic
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # App
    DEBUG: bool = False
    ALLOWED_HOSTS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()  # Reads from .env automatically
```

---

## 3.5 DEPENDENCY INJECTION

### Q: Explain FastAPI's dependency injection system.

**Simple Explanation:**
Dependencies = helpers that your endpoints need. Like a restaurant — the chef (endpoint) needs ingredients (dependencies). The kitchen manager (FastAPI) automatically gets the ingredients before the chef starts cooking.

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

# DATABASE SESSION DEPENDENCY
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session  # Give the session to the endpoint
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# AUTH DEPENDENCY
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await crud.user.get(db, id=user_id)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Using dependencies
@app.get("/users/me")
async def get_my_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    return current_user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin)  # Only admin can delete
):
    await crud.user.remove(db, id=user_id)
    return {"message": "User deleted"}

# Class-based dependency (for complex dependencies)
class PaginationParams:
    def __init__(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100)
    ):
        self.offset = (page - 1) * page_size
        self.limit = page_size

@app.get("/products")
async def list_products(
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_db)
):
    return await crud.product.get_multi(db, skip=pagination.offset, limit=pagination.limit)
```

---

## 3.6 FASTAPI DATABASE WITH SQLALCHEMY (ASYNC)

```python
# db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Test connections before using
    echo=False,          # Set True for query logging in dev
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Important for async!
)

class Base(DeclarativeBase):
    pass

# models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, index=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    orders = relationship("Order", back_populates="user", lazy="selectin")

# crud/base.py — Generic CRUD
from typing import Generic, TypeVar, Type, Optional, List
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(self, db: AsyncSession, id: int) -> Optional[ModelType]:
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()
    
    async def get_multi(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        result = await db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    async def create(self, db: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        db_obj = self.model(**obj_in.model_dump())
        db.add(db_obj)
        await db.flush()  # Get ID without committing
        await db.refresh(db_obj)
        return db_obj
    
    async def update(
        self, db: AsyncSession, *, db_obj: ModelType, obj_in: UpdateSchemaType
    ) -> ModelType:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj
    
    async def remove(self, db: AsyncSession, *, id: int) -> ModelType:
        obj = await self.get(db, id)
        await db.delete(obj)
        return obj
```

---

## 3.7 FASTAPI SECURITY — JWT

```python
# core/security.py
from datetime import datetime, timedelta
from typing import Optional, Union
from passlib.context import CryptContext
from jose import JWTError, jwt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(
    subject: Union[str, int],
    expires_delta: Optional[timedelta] = None,
    additional_claims: dict = None
) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=60))
    
    to_encode = {
        "exp": expire,
        "iat": datetime.utcnow(),
        "sub": str(subject),
        "type": "access",
    }
    if additional_claims:
        to_encode.update(additional_claims)
    
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(subject: Union[str, int]) -> str:
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

# api/v1/endpoints/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    user = await crud.user.get_by_email(db, email=form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return {
        "access_token": create_access_token(user.id),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }

@router.post("/refresh")
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = int(payload.get("sub"))
        user = await crud.user.get(db, id=user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found")
        
        return {
            "access_token": create_access_token(user_id),
            "token_type": "bearer"
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")
```

---

## 3.8 WSGI vs ASGI

### Q: What is the difference between WSGI and ASGI?

**Simple Explanation:**
- **WSGI** (Web Server Gateway Interface) = Old waiter that handles ONE customer at a time. Serve first customer completely, then go to next.
- **ASGI** (Async Server Gateway Interface) = Modern waiter that can take orders from multiple customers while waiting for the kitchen.

```
WSGI (Synchronous):
Request 1 →──────────── Processing ────────────→ Response 1
Request 2 →                          ────── Processing ──→ Response 2
Request 3 →                                         ─── Processing → Response 3
(Sequential — one at a time)

ASGI (Asynchronous):
Request 1 →──── Start ──→ Waiting for DB... ──────────────→ Response 1
Request 2 →──── Start ──→ Waiting for API... ─────────→ Response 2  
Request 3 →──── Start ──→ Waiting for File... ────→ Response 3
(Concurrent — all start together, complete when ready)
```

| Feature          | WSGI (Gunicorn)         | ASGI (Uvicorn)          |
|------------------|-------------------------|-------------------------|
| Protocol         | HTTP only               | HTTP + WebSocket + HTTP/2 |
| Concurrency      | Multiple processes      | Async coroutines        |
| Use case         | CPU-bound, traditional  | I/O-bound, real-time    |
| Frameworks       | Django, Flask           | FastAPI, Django Channels |
| Memory per conn  | High (per process)      | Low (shared event loop) |
| WebSockets       | Not native              | Native                  |

```python
# WSGI application — Django (traditional)
# gunicorn config.wsgi:application --workers 4

# ASGI application — FastAPI
# uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000

# Production: Gunicorn with Uvicorn workers (best of both)
# gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 3.9 FASTAPI MIDDLEWARE

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.com", "https://admin.myapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware
class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        correlation_id = str(uuid.uuid4())
        
        request.state.correlation_id = correlation_id
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={process_time:.4f}s "
            f"correlation_id={correlation_id}"
        )
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self._calls = {}
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        
        if client_ip not in self._calls:
            self._calls[client_ip] = []
        
        # Remove old calls outside the window
        self._calls[client_ip] = [
            t for t in self._calls[client_ip] if now - t < self.period
        ]
        
        if len(self._calls[client_ip]) >= self.calls:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json"
            )
        
        self._calls[client_ip].append(now)
        return await call_next(request)

app.add_middleware(RequestTimingMiddleware)
# Use Redis-based rate limiting in production (above is in-memory only)
```

---

## 3.10 FASTAPI BACKGROUND TASKS

```python
from fastapi import BackgroundTasks

def send_notification(email: str, message: str):
    """This runs in background after response is sent"""
    # send_email(email, message) — takes time
    logger.info(f"Email sent to {email}")

def cleanup_temp_files(file_path: str):
    import os
    os.remove(file_path)

@app.post("/send-report")
async def send_report(
    email: str,
    background_tasks: BackgroundTasks
):
    # Generate report (quick)
    report_path = generate_report()
    
    # Schedule background tasks
    background_tasks.add_task(send_notification, email, "Your report is ready")
    background_tasks.add_task(cleanup_temp_files, report_path)
    
    # Response sent immediately, tasks run after
    return {"message": "Report generation started, you'll receive an email shortly"}
```

---

## 3.11 EXCEPTION HANDLING IN FASTAPI

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

app = FastAPI()

# Custom exceptions
class AppException(Exception):
    def __init__(self, status_code: int, error_code: str, message: str, details: dict = None):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details or {}

class NotFoundException(AppException):
    def __init__(self, resource: str, id: int):
        super().__init__(
            status_code=404,
            error_code="NOT_FOUND",
            message=f"{resource} with id {id} not found"
        )

class UnauthorizedException(AppException):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(status_code=401, error_code="UNAUTHORIZED", message=message)

# Global exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
            "request_id": getattr(request.state, "correlation_id", None)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred"
            }
        }
    )

# Usage in endpoints
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await crud.user.get(db, id=user_id)
    if not user:
        raise NotFoundException("User", user_id)
    return user
```

---

## 3.12 FASTAPI — TOP 50 INTERVIEW QUESTIONS

### Q1: How does FastAPI validate request data automatically?
**Answer:** Through Pydantic models + Python type hints. When you declare a parameter with a type, FastAPI automatically validates incoming data and returns 422 with details if validation fails.

### Q2: What's the difference between `response_model` and return type annotation?
```python
# response_model — controls what's returned to client (serialization + filtering)
# Return type annotation — just for IDE/mypy

@app.get("/users/{id}", response_model=UserPublic)  # Hides password, token fields
async def get_user(id: int) -> UserFull:            # Internal type hint
    return await get_user_with_all_fields(id)        # Even if UserFull has extra fields,
    # only UserPublic fields are sent to client!
```

### Q3: How do you run startup/shutdown events?
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    await database.connect()
    await redis.initialize()
    logger.info("App started")
    
    yield  # App runs here
    
    # SHUTDOWN
    await database.disconnect()
    await redis.close()
    logger.info("App shutting down")

app = FastAPI(lifespan=lifespan)
```

### Q4: How do you handle file uploads in FastAPI?
```python
from fastapi import File, UploadFile
import aiofiles

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Form(None)
):
    # Validate file
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(400, "Invalid file type")
    
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large")
    
    # Save file asynchronously
    file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return {"filename": file.filename, "path": file_path}
```

### Q5: How do you implement rate limiting in FastAPI?
```python
# Using slowapi (Redis-backed)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/resource")
@limiter.limit("5/minute")  # 5 requests per minute per IP
async def limited_endpoint(request: Request):
    return {"message": "OK"}
```

### Q6: How do you test FastAPI applications?
```python
# conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

@pytest.fixture(scope="function")
async def db_session():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def client(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# Test
@pytest.mark.asyncio
async def test_create_user(client):
    response = await client.post("/users", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "strongpassword123"
    })
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

### Q7: How do you handle WebSockets in FastAPI?
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, room_id: str):
        self.active_connections[room_id].remove(websocket)
    
    async def broadcast(self, message: str, room_id: str):
        for connection in self.active_connections.get(room_id, []):
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await manager.connect(websocket, room_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Message in room {room_id}: {data}", room_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        await manager.broadcast(f"A user left room {room_id}", room_id)
```

---

# ═══════════════════════════════════════════════════
# SECTION 4 — CELERY COMPLETE GUIDE
# ═══════════════════════════════════════════════════

---

## 4.1 WHAT IS CELERY?

**Simple Explanation:**
Celery = a post office. You write a letter (task), drop it in a mailbox (queue/broker), and a postman (Celery worker) delivers it later. You don't wait at the mailbox — you go back to your life.

**Real-world use cases:**
- Send emails after user registration
- Generate PDF reports (takes time)
- Process payment in background
- Send push notifications
- Resize uploaded images
- Sync data with external APIs
- Run scheduled jobs (like cron)

```
Your App → Task → Broker (Redis/RabbitMQ) → Worker → Execute Task
                                                      ↓
                                              Result Backend (Redis)
                                                      ↓
                                              Your App checks result
```

---

## 4.2 CELERY ARCHITECTURE

```
┌─────────────┐       ┌─────────────────┐       ┌──────────────────┐
│   Django/   │       │   Message       │       │  Celery Workers  │
│   FastAPI   │──────→│   Broker        │──────→│  (Multiple)      │
│   App       │       │   (Redis /      │       │                  │
└─────────────┘       │   RabbitMQ)     │       │  Worker 1: Task A│
       ↑              └─────────────────┘       │  Worker 2: Task B│
       │                      ↑                 │  Worker 3: Task C│
       │              ┌───────┴─────────┐       └──────────────────┘
       └──────────────│  Result Backend │              │
                      │  (Redis)        │←─────────────┘
                      └─────────────────┘
                      
Celery Beat (Scheduler) → sends periodic tasks to Broker
```

---

## 4.3 COMPLETE CELERY SETUP

```python
# celery_app/tasks.py — Real production examples

from celery import shared_task, group, chain, chord, Signature
from celery.utils.log import get_task_logger
from celery.exceptions import MaxRetriesExceededError
import time

logger = get_task_logger(__name__)

# BASIC TASK
@shared_task(name="users.send_welcome_email")
def send_welcome_email(user_id: int):
    user = User.objects.get(id=user_id)
    # send actual email
    logger.info(f"Welcome email sent to {user.email}")
    return {"status": "sent", "email": user.email}

# TASK WITH RETRY LOGIC
@shared_task(
    bind=True,
    name="payments.process_payment",
    max_retries=5,
    soft_time_limit=30,  # Raises SoftTimeLimitExceeded after 30s
    time_limit=60,        # SIGKILL after 60s
)
def process_payment(self, payment_id: int):
    try:
        payment = Payment.objects.get(id=payment_id)
        result = stripe.charge(payment.amount, payment.card_token)
        payment.status = "completed"
        payment.save()
        return {"status": "completed", "transaction_id": result.id}
    
    except stripe.RateLimitError as exc:
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
    
    except stripe.CardError as exc:
        # Don't retry card errors (user's fault)
        payment.status = "failed"
        payment.save()
        logger.warning(f"Card error for payment {payment_id}: {exc}")
        return {"status": "failed", "reason": str(exc)}
    
    except Exception as exc:
        logger.error(f"Payment processing failed: {exc}")
        raise self.retry(exc=exc, countdown=60)

# TASK WORKFLOWS

# 1. Chain — tasks run sequentially, output of one is input to next
workflow = chain(
    validate_order.s(order_id),        # Returns validated order data
    calculate_shipping.s(),             # Receives order data
    process_payment.s(),                # Receives order + shipping data
    send_confirmation_email.s(),        # Receives full result
)
result = workflow.delay()

# 2. Group — tasks run in parallel
parallel_notifications = group(
    send_email.s(user_id),
    send_sms.s(user_id),
    send_push_notification.s(user_id),
)
parallel_notifications.delay()

# 3. Chord — run group in parallel, then callback when all done
header = group([
    process_image.s(img_id) for img_id in image_ids
])
callback = save_processed_images.s()
chord(header)(callback)

# MONITORING TASKS
from celery.result import AsyncResult

# Calling tasks
result = send_welcome_email.delay(user_id=123)

# Check status
task_result = AsyncResult(result.id)
print(task_result.state)    # PENDING, STARTED, SUCCESS, FAILURE, RETRY
print(task_result.result)   # Return value or exception

# Wait for result (blocks — use carefully!)
final_result = result.get(timeout=10)  # Wait max 10 seconds
```

---

## 4.4 CELERY BEST PRACTICES

```python
# 1. Use task routing — separate queues for different priorities
CELERY_TASK_ROUTES = {
    'apps.*.tasks.critical_*': {'queue': 'critical'},
    'apps.emails.*': {'queue': 'email'},
    'apps.reports.*': {'queue': 'low_priority'},
}

# Start workers for specific queues
# celery -A config worker -Q critical --concurrency=4
# celery -A config worker -Q email --concurrency=2
# celery -A config worker -Q low_priority --concurrency=1

# 2. Task idempotency — safe to run multiple times
@shared_task
def send_activation_email(user_id: int):
    user = User.objects.get(id=user_id)
    if user.activation_email_sent:
        logger.info(f"Activation email already sent for user {user_id}")
        return  # Idempotent — don't send twice
    
    send_email(user.email, "Activate your account")
    user.activation_email_sent = True
    user.save()

# 3. Avoid passing large objects to tasks — pass IDs instead
# BAD
send_report.delay(user_object)  # Serializes entire object!

# GOOD
send_report.delay(user_id=123)  # Task fetches from DB when it runs

# 4. Monitor with Flower
# pip install flower
# celery -A config flower --port=5555
```

---

# ═══════════════════════════════════════════════════
# SECTION 5 — DATABASE INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════

---

## 5.1 POSTGRESQL DEEP DIVE

### Q: What are database indexes and why do they matter?

**Simple Explanation:**
An index is like a book's index — instead of reading every page to find "Django," you go to the index, find the page number, jump directly there.

```sql
-- Without index: Full table scan — reads every row
SELECT * FROM users WHERE email = 'rahul@example.com';  -- Checks all 1M rows!

-- With index: O(log N) lookup
CREATE INDEX idx_users_email ON users(email);  -- Create index
-- Now lookup is nearly instant even with 1M rows!

-- Django ORM: Add index to model
class User(models.Model):
    email = models.EmailField(db_index=True)  # Single field index
    
    class Meta:
        indexes = [
            models.Index(fields=['email']),                    # Simple index
            models.Index(fields=['last_name', 'first_name']), # Composite index
            models.Index(fields=['created_at']),               # For ordering
        ]
```

**Types of indexes:**
```sql
-- B-Tree (default) — for equality, range, ORDER BY
CREATE INDEX idx_email ON users(email);

-- Hash — for equality ONLY (faster for = queries)
CREATE INDEX idx_hash_email ON users USING hash(email);

-- GIN — for full-text search, arrays, JSONB
CREATE INDEX idx_tags ON articles USING gin(tags);

-- Partial index — only indexes some rows (efficient!)
CREATE INDEX idx_active_users ON users(email) WHERE is_active = true;

-- Covering index — includes extra columns (avoids table lookup)
CREATE INDEX idx_user_covering ON users(email) INCLUDE (name, phone);
```

**When indexes slow things down:**
- INSERT/UPDATE/DELETE — indexes must be updated (overhead)
- Too many indexes on write-heavy tables
- Low-cardinality columns (like `gender` — only M/F/Other — index not useful)

---

## 5.2 TRANSACTIONS AND ACID

### Q: What are ACID properties?

**Simple Explanation (Bank transfer analogy):**
Transferring ₹1000 from Rahul to Priya:
1. Debit Rahul: -₹1000
2. Credit Priya: +₹1000

ACID ensures this happens safely:

| Property       | Meaning                                                    | Example                                      |
|----------------|------------------------------------------------------------|----------------------------------------------|
| **Atomicity**  | All or nothing — partial updates never happen              | Both debit AND credit happen, or neither     |
| **Consistency**| DB always valid before and after                           | Total money before = total money after       |
| **Isolation**  | Concurrent transactions don't interfere                    | Two transfers don't mix up                   |
| **Durability** | Committed data survives crashes                            | Power cut after commit — data is still there |

```python
# Django transactions
from django.db import transaction

@transaction.atomic
def transfer_money(from_id, to_id, amount):
    # All DB operations here are wrapped in a transaction
    # If any exception occurs, ALL changes are rolled back
    
    from_account = Account.objects.select_for_update().get(id=from_id)
    to_account = Account.objects.select_for_update().get(id=to_id)
    
    if from_account.balance < amount:
        raise ValueError("Insufficient balance")
    
    from_account.balance = F('balance') - amount
    to_account.balance = F('balance') + amount
    
    from_account.save(update_fields=['balance'])
    to_account.save(update_fields=['balance'])
    # If we reach here — both saves succeeded and transaction commits
    # select_for_update() → locks rows to prevent race conditions
```

---

## 5.3 QUERY OPTIMIZATION

```sql
-- EXPLAIN ANALYZE — see query execution plan
EXPLAIN ANALYZE SELECT * FROM orders 
WHERE user_id = 123 AND status = 'pending'
ORDER BY created_at DESC;

-- Look for:
-- Seq Scan → no index used (bad for large tables)
-- Index Scan → using index (good!)
-- Nested Loop → might indicate N+1 problem

-- Optimization techniques:
-- 1. Add proper indexes
-- 2. Use specific columns instead of SELECT *
-- 3. Paginate results
-- 4. Use JOINs instead of subqueries where possible
-- 5. Use proper data types (int for IDs, not varchar)

-- Django ORM optimization
# Bad
orders = Order.objects.all()  # Selects all columns
for order in orders:
    print(order.user.email)   # N+1 queries!

# Good
orders = Order.objects.select_related('user').only(
    'id', 'status', 'amount', 'user__email'  # Only needed fields
)
```

---

## 5.4 SQL JOINS (Must Know for Interviews)

```sql
-- INNER JOIN — only matching rows
SELECT u.name, o.amount 
FROM users u 
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN — all users, even those without orders (NULL for order columns)
SELECT u.name, o.amount 
FROM users u 
LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN — all orders, even orphan ones
SELECT u.name, o.amount 
FROM users u 
RIGHT JOIN orders o ON u.id = o.user_id;

-- Django ORM equivalent
# INNER JOIN (default with ForeignKey)
orders = Order.objects.select_related('user').all()

# LEFT JOIN
from django.db.models import OuterRef, Subquery
users = User.objects.annotate(
    latest_order_amount=Subquery(
        Order.objects.filter(user=OuterRef('pk'))
        .order_by('-created_at')
        .values('amount')[:1]
    )
)
```

---

# ═══════════════════════════════════════════════════
# SECTION 6 — SYSTEM DESIGN & ARCHITECTURE
# ═══════════════════════════════════════════════════

---

## 6.1 MONOLITH vs MICROSERVICES

**Simple Explanation:**
- **Monolith** = One big restaurant — kitchen, dining room, staff all in one building. Fast to build, hard to scale parts separately.
- **Microservices** = Food court — each restaurant (service) is independent. User Service, Order Service, Payment Service each deploy separately.

```
Monolith:
┌─────────────────────────────────────┐
│ Django App                          │
│  ┌──────┐  ┌───────┐  ┌─────────┐  │
│  │Users │  │Orders │  │Products │  │
│  └──────┘  └───────┘  └─────────┘  │
│        Single Database              │
└─────────────────────────────────────┘

Microservices:
┌──────────┐  ┌──────────┐  ┌──────────┐
│  User    │  │  Order   │  │ Payment  │
│ Service  │  │ Service  │  │ Service  │
│ (Django) │  │(FastAPI) │  │(FastAPI) │
│ Users DB │  │ Orders DB│  │Payment DB│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     └──────────────┴──────────────┘
          API Gateway / Message Bus
```

**When to use what:**

| Factor              | Monolith       | Microservices  |
|---------------------|----------------|----------------|
| Team size           | Small (1-10)   | Large (10+)    |
| Startup stage       | Early stage    | Scaling stage  |
| Deployment          | Simple         | Complex        |
| Performance         | Lower latency  | Network calls  |
| Scalability         | Scale whole app| Scale per service |
| Tech stack          | Same language  | Each can differ|

---

## 6.2 SCALABLE API DESIGN

```
Client Request
      │
      ▼
Load Balancer (Nginx/HAProxy/AWS ALB)
      │
   ┌──┴──┐
   │     │
   ▼     ▼
Server1  Server2  (Multiple Django/FastAPI instances)
   │     │
   └──┬──┘
      │
   ┌──┴──────────┬─────────────┐
   │             │             │
   ▼             ▼             ▼
Primary DB   Read Replica   Redis Cache
(Writes)     (Reads)        (Hot data)
                │
         Message Queue (Celery/RabbitMQ)
                │
            Workers (Background jobs)
```

---

## 6.3 CACHING STRATEGIES

**Simple Explanation:**
Caching = keeping a copy of frequently requested data closer to the user, so you don't have to go to the "far" database every time.

```python
# 1. Cache-Aside (most common)
def get_product(product_id):
    data = cache.get(f"product:{product_id}")
    if not data:
        data = Product.objects.get(id=product_id)
        cache.set(f"product:{product_id}", data, 300)
    return data

# 2. Write-Through — write to cache and DB at same time
def update_product(product_id, data):
    product = Product.objects.get(id=product_id)
    product.update(**data)
    cache.set(f"product:{product_id}", product, 300)  # Update cache too

# 3. Write-Behind — write to cache, then async to DB
# More complex but very fast writes

# 4. Read-Through — cache sits in front of DB (Redis as read-through cache)

# Cache invalidation strategies:
# TTL (time-to-live) — cache expires automatically
# Event-driven — invalidate when data changes (signals, hooks)
# Cache versioning — bump version to invalidate all
```

---

## 6.4 API GATEWAY PATTERN

```python
# In microservices, API Gateway handles:
# - Authentication
# - Rate limiting
# - Routing
# - Load balancing
# - SSL termination
# - Request/response transformation
# - Logging

# Example: nginx as API gateway
# nginx.conf
"""
server {
    listen 80;
    
    location /api/users/ {
        proxy_pass http://user-service:8001;
    }
    
    location /api/orders/ {
        proxy_pass http://order-service:8002;
    }
    
    location /api/payments/ {
        proxy_pass http://payment-service:8003;
    }
}
"""
```

---

# ═══════════════════════════════════════════════════
# SECTION 7 — PRODUCTION & DEPLOYMENT
# ═══════════════════════════════════════════════════

---

## 7.1 DOCKER

```dockerfile
# Dockerfile for Django + FastAPI
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies FIRST (layer caching optimization)
COPY requirements/production.txt .
RUN pip install -r production.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Collect static files (Django)
RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Entrypoint
CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "gthread", \
     "--threads", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

```yaml
# docker-compose.yml — Production-like setup
version: '3.9'

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: myapp_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myapp_user -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  
  web:
    build: .
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    environment:
      - DATABASE_URL=postgresql://myapp_user:${DB_PASSWORD}@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - static_files:/app/static
    ports:
      - "8000:8000"
  
  celery:
    build: .
    command: celery -A config worker --loglevel=info --concurrency=4
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://myapp_user:${DB_PASSWORD}@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
  
  celery-beat:
    build: .
    command: celery -A config beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler
    depends_on:
      - db
      - redis
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/conf.d/default.conf
      - static_files:/static
      - ./certbot/conf:/etc/letsencrypt
    depends_on:
      - web

volumes:
  postgres_data:
  redis_data:
  static_files:
```

---

## 7.2 NGINX CONFIGURATION

```nginx
# docker/nginx.conf
upstream django_app {
    server web:8000;
    keepalive 32;  # Persistent connections
}

server {
    listen 80;
    server_name myapp.com www.myapp.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name myapp.com www.myapp.com;
    
    # SSL
    ssl_certificate /etc/letsencrypt/live/myapp.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/myapp.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    # Static files — served directly by Nginx (much faster!)
    location /static/ {
        alias /static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    location /media/ {
        alias /media/;
        expires 30d;
    }
    
    # API requests — proxy to Django/FastAPI
    location / {
        proxy_pass http://django_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }
    
    # Rate limiting zone definition
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
}
```

---

## 7.3 GUNICORN & UVICORN CONFIGURATION

```python
# gunicorn.conf.py — Django production config
import multiprocessing

# Workers = (2 * CPU cores) + 1
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gthread"     # Threads within each worker
threads = 2
worker_connections = 1000
timeout = 120
keepalive = 5
max_requests = 1000          # Restart worker after N requests (memory leak prevention)
max_requests_jitter = 100    # Random jitter to avoid all workers restarting at once

bind = "0.0.0.0:8000"
accesslog = "-"              # Log to stdout
errorlog = "-"
loglevel = "info"

# Preload app for memory efficiency (copy-on-write)
preload_app = True

# For FastAPI with uvicorn workers:
# gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 7.4 CI/CD WITH GITHUB ACTIONS

```yaml
# .github/workflows/deploy.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, staging]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements/testing.txt
      
      - name: Run linting
        run: |
          flake8 .
          black --check .
          isort --check-only .
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql://test_user:test_pass@localhost/test_db
          REDIS_URL: redis://localhost:6379/0
          SECRET_KEY: test-secret-key
        run: |
          pytest --cov=. --cov-report=xml -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and push Docker image
        run: |
          docker build -t myapp:${{ github.sha }} .
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push myapp:${{ github.sha }}
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            cd /app
            docker pull myapp:${{ github.sha }}
            docker-compose up -d --no-deps web
            docker-compose exec web python manage.py migrate --noinput
            docker-compose exec web python manage.py collectstatic --noinput
            echo "Deployment complete!"
```

---

# ═══════════════════════════════════════════════════
# SECTION 8 — CODING STANDARDS & BEST PRACTICES
# ═══════════════════════════════════════════════════

---

## 8.1 CLEAN CODE PRINCIPLES

```python
# 1. MEANINGFUL NAMES
# BAD
def calc(a, b, c):
    return a * b * (1 - c)

# GOOD
def calculate_discounted_price(original_price: float, quantity: int, discount_rate: float) -> float:
    return original_price * quantity * (1 - discount_rate)

# 2. FUNCTIONS SHOULD DO ONE THING
# BAD
def process_user_registration(data):
    # Validate data
    if not data.get('email'):
        raise ValueError("Email required")
    # Hash password
    data['password'] = hash_password(data['password'])
    # Save to DB
    user = User.objects.create(**data)
    # Send email
    send_welcome_email(user.email)
    # Log
    logger.info(f"User created: {user.id}")
    return user

# GOOD — each function has one responsibility
def validate_registration_data(data: dict) -> dict:
    if not data.get('email'):
        raise ValidationError("Email required")
    if not data.get('password') or len(data['password']) < 8:
        raise ValidationError("Password must be at least 8 characters")
    return data

def create_user(data: dict) -> User:
    data = data.copy()
    data['password'] = hash_password(data.pop('password'))
    return User.objects.create(**data)

def register_user(data: dict) -> User:
    validated_data = validate_registration_data(data)
    user = create_user(validated_data)
    send_welcome_email.delay(user.id)
    logger.info(f"User registered: {user.id}")
    return user

# 3. DON'T REPEAT YOURSELF (DRY)
# BAD — same pagination logic in every view
def get_users(request):
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 20))
    offset = (page - 1) * per_page
    ...

def get_products(request):
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 20))
    offset = (page - 1) * per_page
    ...

# GOOD — extract common logic
def get_pagination_params(request) -> tuple[int, int]:
    page = max(1, int(request.GET.get('page', 1)))
    per_page = min(100, max(1, int(request.GET.get('per_page', 20))))
    offset = (page - 1) * per_page
    return offset, per_page
```

---

## 8.2 API DESIGN STANDARDS

```python
# REST API Best Practices

# 1. Use nouns for resources, not verbs
# BAD: /getUsers, /createProduct, /deleteOrder/5
# GOOD: GET /users, POST /products, DELETE /orders/5

# 2. Consistent response format
class APIResponse:
    @staticmethod
    def success(data=None, message="Success", status_code=200):
        return {
            "success": True,
            "message": message,
            "data": data,
            "error": None
        }, status_code
    
    @staticmethod
    def error(message="Error", error_code="UNKNOWN_ERROR", details=None, status_code=400):
        return {
            "success": False,
            "message": message,
            "data": None,
            "error": {
                "code": error_code,
                "details": details
            }
        }, status_code

# 3. API Versioning
# URL versioning (most common): /api/v1/users, /api/v2/users
# Header versioning: Accept: application/vnd.api+json;version=1

# Django URL versioning
urlpatterns = [
    path('api/v1/', include('api.v1.urls')),
    path('api/v2/', include('api.v2.urls')),
]

# 4. HTTP Status Codes — use correctly!
# 200 OK — successful GET, PUT, PATCH
# 201 Created — successful POST
# 204 No Content — successful DELETE
# 400 Bad Request — validation error, malformed request
# 401 Unauthorized — not authenticated
# 403 Forbidden — authenticated but not permitted
# 404 Not Found — resource doesn't exist
# 409 Conflict — duplicate resource
# 422 Unprocessable Entity — validation failed (FastAPI uses this)
# 429 Too Many Requests — rate limited
# 500 Internal Server Error — server bug
# 503 Service Unavailable — maintenance/overload
```

---

## 8.3 PRODUCTION SETTINGS (DJANGO)

```python
# config/settings/production.py
from .base import *
import os

DEBUG = False
SECRET_KEY = os.environ['SECRET_KEY']  # Never hardcode!
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

# Database with connection pooling
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ['DB_NAME'],
        'USER': os.environ['DB_USER'],
        'PASSWORD': os.environ['DB_PASSWORD'],
        'HOST': os.environ['DB_HOST'],
        'PORT': os.environ.get('DB_PORT', '5432'),
        'CONN_MAX_AGE': 60,
        'OPTIONS': {
            'sslmode': 'require',
        },
    }
}

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'apps': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

---

# ═══════════════════════════════════════════════════
# SECTION 9 — REAL INTERVIEW PREPARATION
# ═══════════════════════════════════════════════════

---

## 9.1 HOW TO ANSWER TECHNICAL QUESTIONS

**The STAR format for scenario questions:**
- **S**ituation: What was the context?
- **T**ask: What needed to be done?
- **A**ction: What did YOU do specifically?
- **R**esult: What was the outcome (metrics!)?

**Example:**
Q: "Tell me about a performance problem you solved."

A (STAR):
- **Situation**: Our Django API was taking 8 seconds for the product listing endpoint.
- **Task**: I needed to reduce it to under 500ms without major refactoring.
- **Action**: Used Django Debug Toolbar to identify 147 SQL queries (N+1 problem). Added `select_related` and `prefetch_related`, added indexes on frequently filtered columns, implemented Redis caching for the product list.
- **Result**: API response time dropped from 8s to 180ms (97% improvement). Server load reduced by 60%.

---

## 9.2 SENIOR DEVELOPER — WHAT INTERVIEWERS EXPECT

**What interviewers look for in Senior Backend roles:**

1. **System Design thinking** — "How would you scale this to 10M users?"
2. **Trade-off awareness** — "Why Django over FastAPI here? What are the drawbacks?"
3. **Production experience** — "What's your monitoring setup? How do you debug prod issues at 3 AM?"
4. **Security mindset** — Always mention auth, rate limiting, input validation
5. **Code quality** — Clean, testable, maintainable code
6. **Ownership** — "I'd set up alerts for this" not "someone should monitor this"

---

## 9.3 COMMON INTERVIEWER TRAPS

**Trap 1: "Django is slow, isn't it?"**
Answer: "Django itself isn't slow — poor usage patterns make it slow. With proper query optimization, caching, and async tasks, Django handles millions of requests. Instagram, Pinterest, Disqus all run Django at scale."

**Trap 2: "Why not just use Flask instead of Django for an API?"**
Answer: "Flask gives flexibility but you end up building the same things yourself (auth, ORM, admin, migrations). Django provides these battle-tested. For a team building a product, Django's conventions save significant time. For microservices where you need extreme lightness, FastAPI is better than both."

**Trap 3: "async is always faster, right?"**
Answer: "Not always. async is better for I/O-bound operations (DB queries, API calls). For CPU-bound tasks, it doesn't help and adds complexity. Async also has its own overhead and debugging challenges. The right tool for the right job."

**Trap 4: "How do you test async code?"**
```python
# Always mention pytest-asyncio
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == expected

# Mock async dependencies
from unittest.mock import AsyncMock

async def test_with_mocked_db(mocker):
    mock_db = AsyncMock()
    mock_db.fetch_one.return_value = {"id": 1, "name": "Test"}
    result = await get_user(db=mock_db, user_id=1)
    assert result["name"] == "Test"
```

---

## 9.4 LIVE CODING TIPS

**Before writing code:**
1. Clarify requirements — "Should I handle pagination? What's the expected data scale?"
2. State your approach — "I'll use a class-based view with JWT auth..."
3. Think edge cases out loud — "What if the user doesn't exist? What if the DB is down?"

**While coding:**
1. Write function signature with type hints first
2. Add docstring explaining what it does
3. Handle happy path first, then error cases
4. Add comments for non-obvious logic

**Example live coding question:**
```python
# Q: "Write a Django view to get top 5 bestselling products this month"

from django.db.models import Sum, Count
from django.utils import timezone
from datetime import timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

class TopSellingProductsView(APIView):
    """Get top 5 best-selling products in the current month"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        now = timezone.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        top_products = (
            OrderItem.objects
            .filter(
                order__status='completed',
                order__created_at__gte=month_start,
            )
            .values('product_id', 'product__name', 'product__price')
            .annotate(
                total_sold=Sum('quantity'),
                total_revenue=Sum('total_price'),
                order_count=Count('order', distinct=True),
            )
            .order_by('-total_sold')[:5]
        )
        
        return Response({
            "period": f"{month_start.strftime('%B %Y')}",
            "products": list(top_products)
        })
```

---

## 9.5 SCENARIO-BASED QUESTIONS

**Q: Your API is suddenly responding in 30 seconds. What do you do?**
```
Step 1: CHECK METRICS — Is it all endpoints or one specific endpoint?
Step 2: CHECK LOGS — Look for error patterns, slow query logs
Step 3: CHECK RESOURCES — CPU, Memory, DB connections, Redis connections
Step 4: CHECK DATABASE — pg_stat_activity in PostgreSQL for long-running queries
Step 5: REPRODUCE — Can you reproduce with simpler test?
Step 6: INVESTIGATE — Add query logging, use Django Debug Toolbar
Step 7: FIX — Could be N+1, missing index, connection pool exhaustion, missing cache
Step 8: MONITOR — After fix, watch metrics for improvement
Step 9: POST-MORTEM — Document what happened and how to prevent
```

**Q: You need to migrate 10 million records without downtime. How?**
```python
# Step 1: Add new column (nullable)
# migrations: AddField(null=True, blank=True)

# Step 2: Deploy code that writes to BOTH old and new column
# Step 3: Run background migration to backfill new column
@shared_task
def backfill_new_column(batch_size=1000):
    last_id = 0
    while True:
        batch = User.objects.filter(
            id__gt=last_id,
            new_field__isnull=True  # Only unprocessed
        ).order_by('id')[:batch_size]
        
        if not batch:
            break
        
        for user in batch:
            user.new_field = transform(user.old_field)
        
        User.objects.bulk_update(batch, ['new_field'])
        last_id = batch.last().id
        
        # Be gentle — don't slam the DB
        time.sleep(0.1)

# Step 4: Once backfill complete, make new column non-nullable
# Step 5: Deploy code that only uses new column
# Step 6: Remove old column in next release
```

**Q: How do you handle secrets in production?**
```python
# NEVER put secrets in code or git!

# Use environment variables
import os
SECRET_KEY = os.environ['SECRET_KEY']  # Raises KeyError if not set (good!)
SECRET_KEY = os.environ.get('SECRET_KEY', 'default')  # Dangerous in production!

# Use .env files with python-decouple
from decouple import config, Csv
SECRET_KEY = config('SECRET_KEY')
ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=Csv())
DEBUG = config('DEBUG', default=False, cast=bool)

# Production: Use secret managers
# AWS Secrets Manager, Google Secret Manager, HashiCorp Vault
# These rotate secrets automatically and provide audit logs
```

---

## 9.6 HR / BEHAVIORAL QUESTIONS

**Q: "Why do you want to leave your current company?"**
Good answer: "I'm looking for opportunities to work on larger scale systems and solve more complex backend challenges. I've grown a lot at my current company but feel ready for the next level."

**Q: "Tell me about a conflict with a colleague."**
Good answer (STAR format): Focus on the process, not the drama. Show how you listened, compromised, and focused on the best technical solution for the project.

**Q: "Where do you see yourself in 5 years?"**
Good answer: "I want to grow into a Tech Lead or Principal Engineer role, where I can design systems that scale and mentor junior developers. I'm particularly interested in distributed systems and backend performance."

**Q: "What's your biggest weakness?"**
Good answer: "I sometimes spend too much time optimizing code before the product is validated. I've been working on shipping faster with MVP mindset and iterating based on real usage data."

---

## 9.7 QUESTIONS TO ASK THE INTERVIEWER

These show seniority and genuine interest:

1. "What does the on-call rotation look like? How many production incidents happen per month?"
2. "What's the current test coverage? Is there a culture of writing tests?"
3. "How does the team handle technical debt? Is there allocated time for refactoring?"
4. "What's the deployment process? How long does it take from code merge to production?"
5. "What are the biggest technical challenges the team is facing right now?"
6. "What does a typical sprint look like? How is work estimated and prioritized?"

---

## 9.8 QUICK REVISION CHEAT SHEET

### Django
| Topic | Key Point |
|-------|-----------|
| N+1 Problem | Use select_related (FK) and prefetch_related (M2M) |
| Transactions | @transaction.atomic, select_for_update() |
| Caching | cache.get/set, @cache_page, Redis backend |
| Auth | JWT (simplejwt), SessionAuthentication |
| Signals | post_save, pre_save — use sparingly |
| Celery | .delay(), .apply_async(), @shared_task |
| Middleware | Runs on every request/response |
| F() | Reference DB field value in queries |
| Q() | Complex AND/OR queries |
| annotate() | Add computed field to each object |
| aggregate() | Single summary value |

### FastAPI
| Topic | Key Point |
|-------|-----------|
| Pydantic | All validation, type hints, nested models |
| Depends() | Dependency injection — auth, db, pagination |
| response_model | Controls what's returned, filters extra fields |
| BackgroundTasks | Run after response, for non-critical tasks |
| lifespan | Startup/shutdown events |
| Async | Use for I/O-bound (DB, HTTP). Not CPU-bound |
| Exception handlers | app.exception_handler() for global handling |
| Middleware | BaseHTTPMiddleware, add_middleware() |

### Database
| Topic | Key Point |
|-------|-----------|
| Index | B-Tree default, GIN for JSONB/arrays |
| ACID | Atomic, Consistent, Isolated, Durable |
| N+1 | select_related, prefetch_related, JOIN |
| Transaction | All or nothing, rollback on error |
| Explain | EXPLAIN ANALYZE to see query plan |

### System Design
| Topic | Key Point |
|-------|-----------|
| Scaling | Horizontal (more servers) > Vertical (bigger server) |
| Cache | Redis, cache-aside pattern, TTL + event invalidation |
| Queue | Celery + Redis/RabbitMQ for async tasks |
| Load Balancer | Round-robin, least connections |
| Microservices | Each service = own DB, API communication |

---

## 9.9 IMPORTANT PRODUCTION CHECKLIST

Before any Django/FastAPI project goes to production, verify:

- [ ] DEBUG = False
- [ ] SECRET_KEY from environment variable
- [ ] ALLOWED_HOSTS configured
- [ ] HTTPS/SSL configured
- [ ] Database using connection pooling
- [ ] Redis for caching and Celery broker
- [ ] Celery workers running for background tasks
- [ ] Proper logging to structured format (JSON)
- [ ] Health check endpoint (`/health/`)
- [ ] Sentry or similar for error tracking
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Database migrations tested
- [ ] Static files served via CDN/Nginx
- [ ] Monitoring (CPU, Memory, Response time, Error rate)
- [ ] Backups for database
- [ ] Security headers (CSP, HSTS, X-Frame-Options)
- [ ] API authentication properly secured
- [ ] Sensitive data encrypted at rest

---

# 🎯 FINAL WORDS

You now have everything you need to crack any Senior Django or FastAPI interview.

**The most important things to remember:**

1. **Explain your thinking out loud** — Interviewers care about how you think, not just the answer
2. **Always mention trade-offs** — "X is better when... but Y is better when..."
3. **Use real numbers when possible** — "Reduced query time from 8s to 180ms"
4. **Security and testing** — Always bring these up proactively
5. **Scalability mindset** — Always think "what if this gets 100x traffic?"

**Your daily practice routine (2 weeks before interview):**
- Day 1-3: Python fundamentals + OOP + Design patterns
- Day 4-6: Django ORM + DRF + Views
- Day 7-9: FastAPI + Pydantic + Async
- Day 10-11: Celery + Redis + Caching  
- Day 12-13: System Design + Production deployment
- Day 14: Mock interviews + Review

**Go build it. One endpoint at a time. You've got this! 🚀**

---
*Guide compiled with 15+ years of production Django & FastAPI experience. Last updated 2025.*
