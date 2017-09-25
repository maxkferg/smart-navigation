import graphene
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from .config import GRAPHQL_API


class CarState(graphene.ObjectType):
    """Mutation return value"""
    rotation = graphene.Float()
    throttle = graphene.Float()


class CarMutation(graphene.Mutation):
    """Adjust the car steering and throttle"""

    class Input():
        left = graphene.Float()
        right = graphene.Float()
        throttle = graphene.Float()
        reset = graphene.Boolean()
        train = graphene.Boolean()

    car = graphene.Field(lambda: CarState)

    @staticmethod
    def mutate(root, args, context, info):
        left = args.get('left')
        right = args.get('right')
        throttle = args.get('throttle')
        reset = args.get('reset')
        train = args.get('train')

        if train:
            train_car()
        if reset:
            hw.reset()
        if left:
            hw.steering.turn_left(left)
        if right:
            hw.steering.turn_right(right)
        if throttle:
            hw.throttle.set_throttle(throttle)
        return CarMutation(car=CarState())



class Mutations(graphene.ObjectType):
    controlCar = CarMutation.Field()



class Query(graphene.ObjectType):
    car = graphene.Field(lambda: CarState)
    hello = graphene.String(description='A typical hello world')
    steering = graphene.Float(description='The steering angle of the car')
    throttle = graphene.Float(description='The throttle of the car')

    def resolve_hello(self, args, context, info):
        return 'World'

    def resolve_thottle(self, args, context, info):
        return hw.throttle.get_throttle()

    def resolve_steering(self, args, context, info):
        return hw.steering.get_rotation()

    def resolve_car(self, args, context, info):
        return CarState()



schema = graphene.Schema(query=Query, mutation=Mutations)
transport = RequestsHTTPTransport(GRAPHQL_API, use_json=True)
client = Client(schema=schema, transport=transport)




def get_car_state():
    """Get the current car control parameters"""
    query = gql('''
        query Car {
            car {
              rotation
              throttle
            }
        }
    ''')

    return client.execute(query)


def control_car(rotation, throttle, reset=False):
    """Control the Car"""
    query = gql('''
        mutation Car($right: Float, $throttle: Float $reset: Boolean) {
          controlCar(right: $right, throttle: $throttle, reset: $reset) {
            car {
              rotation
              throttle
            }
          }
        }
    ''')

    print('Sending control signal',(rotation,throttle,reset))

    return client.execute(query, {
        'right': float(rotation),
        'throttle': float(throttle),
        'reset': bool(reset)
    })


