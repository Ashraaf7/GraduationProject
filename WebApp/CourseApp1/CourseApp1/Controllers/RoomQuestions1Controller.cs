using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using CourseApp1.data;
using CourseApp1.ViewModel;
using Microsoft.AspNet.Identity;


namespace CourseApp1.Controllers
{
    public class RoomQuestions1Controller : Controller
    {
        private courses_dbEntities db = new courses_dbEntities();

        // GET: RoomQuestions1
        public ActionResult Index()
        {
            return View(db.Rooms.ToList());
        }
        public ActionResult Questions(int? id, string search)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var roomQuestions = db.RoomQuestions.Include(r => r.Room).Include(r => r.AspNetUser).Where(t => t.RoomId == id);
            //ViewBag.RoomId = id;
            if (!String.IsNullOrEmpty(search))
            {
                roomQuestions = roomQuestions.Where(s => s.Content.Contains(search));
            }
            string room = db.Rooms.Where(r => r.RoomId == id).Select(r => r.RoomName).FirstOrDefault();
            ViewBag.RoomName = room;
            ViewBag.RoomId = new SelectList(db.Rooms, "RoomId", "RoomName");
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email");
            ViewBag.Questions = roomQuestions.OrderByDescending(i=>i.CreationDate).ToList();
            return View();
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        [Authorize]
        public ActionResult Questions([Bind(Include = "QuestionId,Content,RoomId")] RoomQuestion roomQuestion, int? id)
        {
            if (id != null)
            {
                roomQuestion.RoomId = (int)id;
            }
            roomQuestion.UserId = User.Identity.GetUserId();
            roomQuestion.CreationDate = DateTime.Now;
            roomQuestion.IsAnswered = false;
            if (ModelState.IsValid)
            {
                db.RoomQuestions.Add(roomQuestion);
                db.SaveChanges();
                return RedirectToAction("Questions", "RoomQuestions1", new {id=id});
            }

            ViewBag.RoomId = new SelectList(db.Rooms, "RoomId", "RoomName", roomQuestion.RoomId);
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email", roomQuestion.UserId);
            return View(roomQuestion);
        }

        public ActionResult Answers(int? id, string search)
        {
            if (id == null)
            {
                return new HttpStatusCodeResult(HttpStatusCode.BadRequest);
            }
            var answers = db.Answers.Include(a => a.RoomQuestion).Include(a => a.AspNetUser).Where(t => t.QuestionId == id);
            if (!String.IsNullOrEmpty(search))
            {
                answers = answers.Where(s => s.Content.Contains(search));
            }
            string question = db.RoomQuestions.Where(s => s.QuestionId == id).Select(s => s.Content).FirstOrDefault();
            string user = db.RoomQuestions.Where(s => s.QuestionId == id).Select(s => s.AspNetUser.Email).FirstOrDefault();
            var date = db.RoomQuestions.Where(s => s.QuestionId == id).Select(s => s.CreationDate).FirstOrDefault();
            ViewBag.Question = question;
            ViewBag.User = user;
            ViewBag.Date = date;

            int Rid=db.RoomQuestions.Where(s=>s.QuestionId==id).Select(s=>s.RoomId).FirstOrDefault();
            ViewBag.RoomId = Rid;
            ViewBag.QuestionId = new SelectList(db.RoomQuestions, "QuestionId", "Content");
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email");
            ViewBag.Answers = answers.OrderByDescending(i => i.CreationDate).ToList();
            return View();
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        [Authorize]
        public ActionResult Answers([Bind(Include = "AnswerId,Content,QuestionId,UserId")] Answer answer, int? id)
        {
            if (id != null)
            {
                answer.QuestionId = (int)id;
            }                
            answer.UserId = User.Identity.GetUserId();
            answer.CreationDate = DateTime.Now;
            if (ModelState.IsValid)
            {
                db.Answers.Add(answer);
                db.SaveChanges();
                return RedirectToAction("Answers", "RoomQuestions1", new { id = id });
            }

            ViewBag.QuestionId = new SelectList(db.RoomQuestions, "QuestionId", "Content", answer.QuestionId);
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email", answer.UserId);
            return View(answer);
        }

        public ActionResult Add()
        {
            ViewBag.RoomId = new SelectList(db.Rooms, "RoomId", "RoomName");
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email");
            return PartialView("_Create");
        }

        // GET: RoomQuestions1/Create
        public ActionResult Create()
        {
            ViewBag.RoomId = new SelectList(db.Rooms, "RoomId", "RoomName");
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email");
            return View();
        }

        // POST: RoomQuestions1/Create
        // To protect from overposting attacks, enable the specific properties you want to bind to, for 
        // more details see https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "QuestionId,Content,RoomId,UserId,CreationDate,IsAnswered")] RoomsViewModel vm)
        {
            
            RoomQuestion roomQuestion = new RoomQuestion();
            roomQuestion = vm.roomQuestion;
            if (ModelState.IsValid)
            {
                db.RoomQuestions.Add(roomQuestion);
                db.SaveChanges();
                return RedirectToAction("Index");
            }

            ViewBag.RoomId = new SelectList(db.Rooms, "RoomId", "RoomName", roomQuestion.RoomId);
            ViewBag.UserId = new SelectList(db.AspNetUsers, "Id", "Email", roomQuestion.UserId);
            return View(roomQuestion);
        }        

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
