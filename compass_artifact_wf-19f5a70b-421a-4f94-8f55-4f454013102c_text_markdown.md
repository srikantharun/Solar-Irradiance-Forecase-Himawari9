# FastAPI: From fundamentals to production-ready

This comprehensive guide examines modern FastAPI projects that demonstrate clear understanding of Starlette and Pydantic dependencies. Whether you're preparing for interviews or building production applications, these examples will help you master FastAPI's core architecture.

## Introduction to FastAPI: Getting started with the fundamentals

FastAPI builds on two powerful foundations: **Starlette** (the web framework) and **Pydantic** (the data validation library). Understanding how these dependencies work together is key to mastering FastAPI.

The [official FastAPI repository](https://github.com/tiangolo/fastapi) provides excellent examples demonstrating core concepts, from basic endpoints to advanced techniques. The tutorial section progressively introduces features with clear explanations and interactive examples.

For dependency injection, a fundamental FastAPI concept:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return commons
```

This simple pattern is tremendously powerful – the `Depends` function injects dependencies (like database connections, authentication, or shared utilities) into your route functions.

For more structured learning, [zhanymkanov/fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices) offers production-based best practices with a focus on advanced Starlette and Pydantic usage. It covers patterns like custom base models for standardized responses:

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict

def datetime_to_gmt_str(dt: datetime) -> str:
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")

class CustomModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={datetime: datetime_to_gmt_str},
        populate_by_name=True,
    )
    
    def serializable_dict(self, **kwargs):
        """Return a dict which contains only serializable fields."""
        default_dict = self.model_dump()
        return jsonable_encoder(default_dict)
```

For more complete project structures, [iam-abbas/FastAPI-Production-Boilerplate](https://github.com/iam-abbas/FastAPI-Production-Boilerplate) demonstrates a scalable architecture following layered design principles.

## Authentication implementations that showcase dependency injection

FastAPI's dependency injection system shines with authentication. Modern authentication implementations leverage Pydantic for data validation and Starlette's security utilities.

The [k4black/fastapi-jwt](https://github.com/k4black/fastapi-jwt) project implements JWT authentication with clean dependency injection:

```python
from fastapi import FastAPI, Security, Response
from fastapi_jwt import JwtAuthorizationCredentials, JwtAccessBearer

app = FastAPI()
access_security = JwtAccessBearer(secret_key="secret_key", auto_error=True)

@app.post("/auth")
def auth():
    subject = {"username": "username", "role": "user"}
    return {"access_token": access_security.create_access_token(subject=subject)}

@app.get("/users/me")
def read_current_user(
    credentials: JwtAuthorizationCredentials = Security(access_security),
):
    return {"username": credentials["username"], "role": credentials["role"]}
```

For a complete user management system, [fastapi-users](https://github.com/fastapi-users/fastapi-users) offers a sophisticated approach with multiple authentication backends (JWT, cookie, OAuth social auth):

```python
from fastapi import Depends, FastAPI
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

app = FastAPI()

# Create authentication backend
auth_backend = JWTAuthentication(secret="SECRET", lifetime_seconds=3600)

# Create FastAPIUsers instance
fastapi_users = FastAPIUsers(
    user_db,
    [auth_backend],
    User,
    UserCreate,
    UserUpdate,
    UserDB,
)

# Add auth routes
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

# Protect routes with dependencies
@app.get("/protected-route")
def protected_route(user = Depends(fastapi_users.current_user(active=True))):
    return {"message": f"Hello, {user.email}"}
```

For role-based access control, [Kludex/fastapi-authorization](https://github.com/Kludex/fastapi-authorization) provides a clean RBAC implementation:

```python
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from fastapi_authorization.rbac import RBAC

class Token(BaseModel):
    role: str

def get_token():
    return Token(role="admin")

def role_callback(token: Token = Depends(get_token)) -> str:
    return token.role

auth = RBAC(role_callback, roles=["admin"])
auth.add_role("admin", permissions=["read:user"])

app = FastAPI()

@app.get("/", dependencies=[auth.Permission("read:user")])
def get_user():
    return {"Hello": "World"}
```

Session-based authentication is demonstrated in [jordanisaacs/fastapi-sessions](https://github.com/jordanisaacs/fastapi-sessions), showing how to manage cookie-based sessions with Pydantic validation.

## Asynchronous operations: The heart of modern FastAPI

FastAPI's asynchronous capabilities leverage Starlette's async foundations. Several projects showcase effective async patterns.

The [igorbenav/FastAPI-boilerplate](https://github.com/igorbenav/FastAPI-boilerplate) demonstrates async database operations with SQLAlchemy 2.0:

```python
@router.get("/entities", response_model=PaginatedListResponse[EntityRead])
async def read_entities(
    request: Request, 
    db: Annotated[AsyncSession, Depends(async_get_db)],
    page: int = 1, 
    items_per_page: int = 10
):
    entities_data = await crud_entity.get_multi(
        db=db,
        offset=compute_offset(page, items_per_page),
        limit=items_per_page,
        schema_to_select=UserRead,
        is_deleted=False,
    )
    return paginated_response(crud_data=entities_data, page=page, items_per_page=items_per_page)
```

For WebSocket implementations, [DontPanicO/fastapi-distributed-websocket](https://github.com/DontPanicO/fastapi-distributed-websocket) provides a robust solution:

```python
@app.websocket('/ws/{conn_id}')
async def websocket_endpoint(
    ws: WebSocket,
    conn_id: str,
    *,
    topic: Optional[Any] = None,
) -> None:
    connection: Connection = await manager.new_connection(ws, conn_id)
    # Preferred way of handling WebSocketDisconnect
    async for msg in connection.iter_json():
        await manager.receive(connection, msg)
    await manager.remove_connection(connection)
```

The [permitio/fastapi_websocket_pubsub](https://github.com/permitio/fastapi_websocket_pubsub) project implements pub/sub patterns over WebSockets:

```python
import asyncio
from fastapi import FastAPI
from fastapi_websocket_pubsub import PubSubEndpoint

app = FastAPI()
# Init endpoint
endpoint = PubSubEndpoint()
# Register the endpoint on the app
endpoint.register_route(app, "/pubsub")

# Register a regular HTTP route
@app.get("/trigger")
async def trigger_events():
    # Upon request trigger an event
    endpoint.publish(["triggered"])
```

Understanding when to use async is critical, as explained in [zhanymkanov/fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices):

```python
import asyncio
import time
from fastapi import APIRouter

router = APIRouter()

@router.get("/terrible-ping")
async def terrible_ping():
    time.sleep(10)  # I/O blocking operation, blocks the whole process
    return {"pong": True}

@router.get("/good-ping")
def good_ping():
    time.sleep(10)  # I/O blocking operation in a separate thread
    return {"pong": True}

@router.get("/perfect-ping")
async def perfect_ping():
    await asyncio.sleep(10)  # non-blocking I/O operation
    return {"pong": True}
```

Background tasks and job queues integrate well with FastAPI's async foundation, as shown in this ARQ implementation:

```python
# Define an async background task
async def sample_background_task(ctx, name: str) -> str:
    await asyncio.sleep(5)
    return f"Task {name} is complete!"

# Endpoint that enqueues a background task
@router.post("/task", response_model=Job, status_code=201)
async def create_task(message: str):
    job = await queue.pool.enqueue_job("sample_background_task", message)
    return {"id": job.job_id}
```

## API gateway integrations for scalable deployments

FastAPI applications can be deployed behind API gateways for enhanced security, routing, and scaling capabilities.

The [patheard/aws-fastapi-lambda](https://github.com/patheard/aws-fastapi-lambda) project demonstrates integrating FastAPI with AWS API Gateway and Lambda using Mangum:

```python
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

handler = Mangum(app)
```

For custom API gateway implementation, [dotX12/fastapi-gateway](https://github.com/dotX12/fastapi-gateway) provides a decorator-based approach for service routing:

```python
from fastapi import FastAPI, Depends
from starlette.requests import Request
from starlette.responses import Response
from fastapi_gateway import route
from pydantic import BaseModel

app = FastAPI()
SERVICE_URL = "http://microservice.example.com:8002"

class DataModel(BaseModel):
    example_int: int
    example_str: str

@route(
    request_method=app.post,
    service_url=SERVICE_URL,
    gateway_path='/path/{path_param}',
    service_path='/v1/service_path/{path_param}',
    query_params=['query_param'],
    body_params=['data_model'],
)
async def gateway_endpoint(
    path_param: int, 
    query_param: str,
    data_model: DataModel, 
    request: Request, 
    response: Response
):
    pass
```

For Kong integration, projects like [AliBigdeli/FastApi-GRPC-Todo-Microservice-App](https://github.com/AliBigdeli/FastApi-GRPC-Todo-Microservice-App) demonstrate declarative configuration:

```yaml
_format_version: "2.1"
_transform: true

services:
  - name: todo-api
    url: http://todo_api:8000
    routes:
      - name: todo-api-route
        paths:
          - /todo
    plugins:
      - name: cors
```

## TypeScript integration for end-to-end type safety

One of FastAPI's strengths is its ability to generate OpenAPI schemas that can be used to create type-safe TypeScript clients.

The [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template) provides a complete implementation for TypeScript integration.

For converting Pydantic models to TypeScript interfaces, tools like [pydantic-to-typescript](https://github.com/phillipdupuis/pydantic-to-typescript) ensure type consistency:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Profile(BaseModel):
    username: str
    age: Optional[int]
    hobbies: List[str]

class LoginResponseData(BaseModel):
    token: str
    profile: Profile

@app.post('/login/', response_model=LoginResponseData)
def login(body: LoginCredentials):
    profile = Profile(username=body.username, age=72, hobbies=['coding'])
    return LoginResponseData(token='token-value', profile=profile)
```

This generates corresponding TypeScript interfaces:

```typescript
export interface Profile {
  username: string;
  age?: number;
  hobbies: string[];
}

export interface LoginResponseData {
  token: string;
  profile: Profile;
}
```

Using the OpenAPI Generator with FastAPI produces comprehensive TypeScript clients:

```python
from fastapi import FastAPI
from fastapi.routing import APIRoute

def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """Simplify operation IDs so that generated client function names are cleaner"""
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

app = FastAPI()

@app.get("/items/")
async def get_items():
    return [{"name": "Item 1"}, {"name": "Item 2"}]

# Call after all routes have been added
use_route_names_as_operation_ids(app)
```

The resulting TypeScript client enables strongly-typed API calls:

```typescript
import { Configuration, ItemsApi } from './generated-client';

const config = new Configuration({
  basePath: 'http://localhost:8000'
});

const itemsApi = new ItemsApi(config);

// TypeScript autocomplete will show available methods and types
async function getItems() {
  const response = await itemsApi.getItems();
  console.log(response.data);
}
```

## Testing approaches and best practices

Modern FastAPI applications require robust testing strategies. The [nsidnev/fastapi-realworld-example-app](https://github.com/nsidnev/fastapi-realworld-example-app) demonstrates comprehensive testing with pytest:

```python
@pytest.fixture
def app() -> FastAPI:
    from app.main import get_application  # local import for testing purpose
    return get_application()

@pytest.fixture
async def initialized_app(app: FastAPI) -> FastAPI:
    async with LifespanManager(app):
        app.state.pool = await FakeAsyncPGPool.create_pool(app.state.pool)
        yield app

@pytest.fixture
async def client(initialized_app: FastAPI) -> AsyncClient:
    async with AsyncClient(
        app=initialized_app,
        base_url="http://testserver",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client
```

For unit testing Pydantic models:

```python
import pytest
from pydantic import ValidationError
from app.models import User

def test_user_model_valid_data():
    user_data = {
        "id": 1,
        "email": "user@example.com",
        "username": "testuser",
        "password": "securepassword"
    }
    user = User(**user_data)
    assert user.id == 1
    assert user.email == "user@example.com"

def test_user_model_invalid_email():
    user_data = {
        "id": 1,
        "email": "invalid-email",  # Invalid email format
        "username": "testuser",
        "password": "securepassword"
    }
    with pytest.raises(ValidationError):
        User(**user_data)
```

For testing async endpoints:

```python
@pytest.mark.anyio
async def test_read_item_endpoint_async():
    async with AsyncClient(
        transport=ASGITransport(app=app), 
        base_url="http://test"
    ) as ac:
        response = await ac.get("/items/1", headers={"X-Token": "valid_token"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
```

Dependency overrides are crucial for isolation in testing:

```python
# In your test setup
app.dependency_overrides[get_current_user] = lambda: test_user
app.dependency_overrides[get_db] = get_test_db

# After your test
app.dependency_overrides = {}
```

Key testing best practices include:
1. Use dependency injection to make code testable
2. Isolate tests using fixtures with appropriate scopes
3. Mock external services to avoid side effects
4. Use in-memory databases for faster test execution
5. Test both happy paths and error cases

## SQLAlchemy integration for database operations

Modern FastAPI applications frequently use SQLAlchemy 2.0 for database access. The [rhoboro/async-fastapi-sqlalchemy](https://github.com/rhoboro/async-fastapi-sqlalchemy) project demonstrates async SQLAlchemy with FastAPI.

The standard pattern for async SQLAlchemy sessions involves:

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import AsyncGenerator
from fastapi import Depends

# Create engine and session factory
async_engine = create_async_engine(
    "postgresql+asyncpg://user:password@localhost/dbname",
    echo=True,  # For SQL logging
    pool_pre_ping=True,  # Health check for connections
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    expire_on_commit=False,  # Important for async sessions
    future=True,
)

# Dependency for FastAPI
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
```

Modern SQLAlchemy 2.0 models use a more type-annotated approach:

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey
from typing import List, Optional

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    
    # Relationship
    posts: Mapped[List["Post"]] = relationship(back_populates="author")
```

Many projects implement a repository pattern to abstract database operations:

```python
class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalars().first()
    
    async def get_all(self) -> List[User]:
        result = await self.session.execute(select(User))
        return result.scalars().all()
    
    async def create(self, user_data: UserCreate) -> User:
        user = User(**user_data.dict())
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
```

For complex query building:

```python
async def search_posts(
    self,
    keyword: Optional[str] = None,
    category_id: Optional[int] = None,
    published_only: bool = True,
    offset: int = 0,
    limit: int = 20
) -> List[Post]:
    query = select(Post)
    
    if keyword:
        query = query.where(
            or_(
                Post.title.contains(keyword),
                Post.content.contains(keyword)
            )
        )
    
    if category_id:
        query = query.where(Post.category_id == category_id)
    
    if published_only:
        query = query.where(Post.is_published == True)
    
    query = query.offset(offset).limit(limit).order_by(Post.created_at.desc())
    
    result = await self.session.execute(query)
    return result.scalars().all()
```

For interview preparation, be sure to understand:
1. How FastAPI builds on Starlette for the web framework functionality
2. How Pydantic handles data validation and serialization
3. How dependency injection works and why it's powerful
4. When to use async vs sync functions
5. Best practices for structuring larger applications
6. How to properly test FastAPI applications

The projects highlighted in this report demonstrate clear understanding of these core concepts and provide excellent reference implementations for real-world applications.

Whether you're just starting with FastAPI or preparing for advanced interviews, these modern implementations will help you grasp both the fundamentals and production-ready patterns of this powerful framework.